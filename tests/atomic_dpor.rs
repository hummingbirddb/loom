#![deny(warnings, rust_2018_idioms)]

//! Models whose failing execution needs a store or RMW to be reordered ahead
//! of a load that is not the atomic's most recent access.

use loom::sync::atomic::AtomicUsize;
use loom::sync::Arc;
use loom::thread;

use std::collections::BTreeSet;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::Mutex;

const EMPTY: usize = 0;
const READY: usize = 1;

/// A slot that is published before its value is written. The writer checks
/// the state with a load before its compare-and-swap, so the writer's own
/// load is the state's last access when the compare-and-swap is scheduled.
/// The failing execution runs the compare-and-swap before the reader's load.
#[test]
#[should_panic(expected = "read an unwritten slot")]
fn load_then_compare_exchange() {
    loom::model(|| {
        let state = Arc::new(AtomicUsize::new(EMPTY));
        let value = Arc::new(AtomicUsize::new(0));

        let reader = {
            let (state, value) = (state.clone(), value.clone());
            thread::spawn(move || {
                if state.load(SeqCst) == READY {
                    assert_eq!(value.load(SeqCst), 7, "read an unwritten slot");
                }
            })
        };

        let writer = thread::spawn(move || {
            if state.load(SeqCst) == EMPTY
                && state.compare_exchange(EMPTY, READY, SeqCst, SeqCst).is_ok()
            {
                // The bug: the value is written after the slot is published.
                value.store(7, SeqCst);
            }
        });

        reader.join().unwrap();
        writer.join().unwrap();
    });
}

/// Two threads read a counter before a third thread resets it. The second
/// reader starts the resetter, so its load is ordered before the reset and
/// is the counter's last access when the reset is scheduled. The failing
/// execution runs the reset before the first reader's load.
#[test]
#[should_panic(expected = "sampled a reset counter")]
fn two_loads_then_store() {
    loom::model(|| {
        let counter = Arc::new(AtomicUsize::new(1));

        let sampler = {
            let counter = counter.clone();
            thread::spawn(move || {
                // The bug: nothing orders this load before the reset.
                assert_eq!(counter.load(SeqCst), 1, "sampled a reset counter");
            })
        };

        let supervisor = thread::spawn(move || {
            let seen = counter.load(SeqCst);
            let resetter = thread::spawn(move || counter.store(0, SeqCst));
            resetter.join().unwrap();
            seen
        });

        sampler.join().unwrap();
        assert_eq!(supervisor.join().unwrap(), 1);
    });
}

/// One step of a thread in `explores_every_interleaving`.
#[derive(Clone, Copy, Debug)]
enum Op {
    /// Load the atomic into the thread's register.
    Load,
    /// Store the constant.
    Store(usize),
    /// Store the thread's register plus one.
    StoreNext,
    /// Compare-and-swap from the thread's register to the register plus one,
    /// and load the current value into the register.
    CompareExchange,
    /// Add to the atomic.
    FetchAdd(usize),
}

const OPS: [Op; 5] = [
    Op::Load,
    Op::Store(5),
    Op::StoreNext,
    Op::CompareExchange,
    Op::FetchAdd(2),
];

trait Cell {
    fn load(&self) -> usize;
    fn store(&self, value: usize);
    fn compare_exchange(&self, current: usize, new: usize) -> Result<usize, usize>;
    fn fetch_add(&self, value: usize) -> usize;
}

impl Cell for AtomicUsize {
    fn load(&self) -> usize {
        AtomicUsize::load(self, SeqCst)
    }

    fn store(&self, value: usize) {
        AtomicUsize::store(self, value, SeqCst)
    }

    fn compare_exchange(&self, current: usize, new: usize) -> Result<usize, usize> {
        AtomicUsize::compare_exchange(self, current, new, SeqCst, SeqCst)
    }

    fn fetch_add(&self, value: usize) -> usize {
        AtomicUsize::fetch_add(self, value, SeqCst)
    }
}

impl Cell for std::cell::Cell<usize> {
    fn load(&self) -> usize {
        self.get()
    }

    fn store(&self, value: usize) {
        self.set(value)
    }

    fn compare_exchange(&self, current: usize, new: usize) -> Result<usize, usize> {
        let value = self.get();
        if value == current {
            self.set(new);
            Ok(value)
        } else {
            Err(value)
        }
    }

    fn fetch_add(&self, value: usize) -> usize {
        self.replace(self.get() + value)
    }
}

fn step(op: Op, cell: &impl Cell, register: &mut usize) {
    match op {
        Op::Load => *register = cell.load(),
        Op::Store(value) => cell.store(value),
        Op::StoreNext => cell.store(*register + 1),
        Op::CompareExchange => match cell.compare_exchange(*register, *register + 1) {
            Ok(value) | Err(value) => *register = value,
        },
        Op::FetchAdd(value) => *register = cell.fetch_add(value),
    }
}

/// The outcomes of every interleaving of `threads`, by enumeration: the
/// atomic's final value and each thread's final register.
fn enumerate(threads: &[Vec<Op>]) -> BTreeSet<Vec<usize>> {
    fn go(
        threads: &[Vec<Op>],
        pcs: &mut Vec<usize>,
        value: usize,
        registers: &[usize],
        outcomes: &mut BTreeSet<Vec<usize>>,
    ) {
        let mut done = true;

        for i in 0..threads.len() {
            if let Some(&op) = threads[i].get(pcs[i]) {
                done = false;

                let cell = std::cell::Cell::new(value);
                let mut registers = registers.to_vec();
                step(op, &cell, &mut registers[i]);

                pcs[i] += 1;
                go(threads, pcs, cell.get(), &registers, outcomes);
                pcs[i] -= 1;
            }
        }

        if done {
            let mut outcome = vec![value];
            outcome.extend_from_slice(registers);
            outcomes.insert(outcome);
        }
    }

    let mut outcomes = BTreeSet::new();
    let registers = vec![0; threads.len()];
    go(
        threads,
        &mut vec![0; threads.len()],
        0,
        &registers,
        &mut outcomes,
    );
    outcomes
}

/// The outcomes of the executions that the model checker explores.
fn explore(threads: &[Vec<Op>]) -> BTreeSet<Vec<usize>> {
    let outcomes = std::sync::Arc::new(Mutex::new(BTreeSet::new()));
    let threads = threads.to_vec();

    {
        let outcomes = outcomes.clone();
        loom::model(move || {
            let cell = Arc::new(AtomicUsize::new(0));

            let handles: Vec<_> = threads
                .iter()
                .map(|ops| {
                    let (cell, ops) = (cell.clone(), ops.clone());
                    thread::spawn(move || {
                        let mut register = 0;
                        for op in ops {
                            step(op, &*cell, &mut register);
                        }
                        register
                    })
                })
                .collect();

            let mut outcome = vec![0];
            for handle in handles {
                outcome.push(handle.join().unwrap());
            }
            outcome[0] = cell.load(SeqCst);

            outcomes.lock().unwrap().insert(outcome);
        });
    }

    let outcomes = outcomes.lock().unwrap().clone();
    outcomes
}

/// Checks that the model checker reaches every outcome that an interleaving
/// produces, for every program of two threads with two operations each, and
/// for the programs of three threads with one load, store, or
/// compare-and-swap each. The model also admits outcomes that no
/// interleaving produces, which this does not check.
#[test]
fn explores_every_interleaving() {
    let mut programs = Vec::new();

    for &a in &OPS {
        for &b in &OPS {
            for &c in &OPS {
                for &d in &OPS {
                    programs.push(vec![vec![a, b], vec![c, d]]);
                }
            }
        }
    }

    let ops = [Op::Load, Op::StoreNext, Op::CompareExchange];

    for &a in &ops {
        for &b in &ops {
            for &c in &ops {
                programs.push(vec![vec![a], vec![b], vec![c]]);
            }
        }
    }

    for threads in &programs {
        let missed: Vec<_> = enumerate(threads)
            .difference(&explore(threads))
            .cloned()
            .collect();

        assert!(
            missed.is_empty(),
            "{:?} never produced {:?}",
            threads,
            missed
        );
    }
}
