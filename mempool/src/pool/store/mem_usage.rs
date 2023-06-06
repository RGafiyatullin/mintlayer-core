// Copyright (c) 2022 RBB S.r.l
// opensource@mintlayer.org
// SPDX-License-Identifier: MIT
// Licensed under the MIT License;
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://github.com/mintlayer/mintlayer-core/blob/master/LICENSE
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Estimate and track memory usage taken by data structures.

use std::{
    mem,
    sync::atomic::{AtomicUsize, Ordering},
};

use common::chain::{
    signature::inputsig::InputWitness, stakelock::StakePoolData, SignedTransaction, TxInput,
    TxOutput,
};

use super::TxMempoolEntry;

/// Structure that stores the current memory usage and keeps track of its changes
#[derive(Debug)]
pub struct Tracker {
    current_usage: AtomicUsize,
}

impl Tracker {
    pub fn new() -> Self {
        let current_usage = AtomicUsize::new(0);
        Self { current_usage }
    }

    pub fn get_usage(&self) -> usize {
        self.current_usage.load(Ordering::Acquire)
    }

    fn add(&self, amount: usize) {
        self.current_usage.fetch_add(amount, Ordering::AcqRel);
    }

    fn sub(&self, amount: usize) {
        self.current_usage.fetch_sub(amount, Ordering::AcqRel);
    }
}

/// A data structure which has its memory consumption tracked
#[must_use = "Memory-tracked object dropped without using Tracked::release"]
#[derive(Eq, PartialEq, PartialOrd, Ord, Debug)]
pub struct Tracked<T, D> {
    obj: T,
    drop_policy: D,
}

impl<T: ZeroUsageDefault, D: Default> Default for Tracked<T, D> {
    fn default() -> Self {
        let obj = T::default();
        assert_eq!(
            obj.indirect_memory_usage(),
            0,
            "Default not zero-size despite being marked as such"
        );

        let drop_policy = D::default();
        Self { obj, drop_policy }
    }
}

impl<T: MemoryUsage, D: DropPolicy + Default> Tracked<T, D> {
    /// Create a new object with tracked memory usage
    pub fn new(tracker: &Tracker, obj: T) -> Self {
        tracker.add(obj.indirect_memory_usage());
        let drop_policy = D::default();
        Self { obj, drop_policy }
    }

    /// Release the object from the tracker and return it as a value
    pub fn release(this: Self, tracker: &Tracker) -> T {
        tracker.sub(this.obj.indirect_memory_usage());
        Self::forget(this)
    }

    /// Forget about the object being tracked without updating the tracker. Useful during tear down
    /// when the tracker is no longer in use.
    pub fn forget(mut this: Self) -> T {
        this.drop_policy.on_release();
        this.obj
    }

    /// Get mutable access to the tracked object.
    ///
    /// Memory usage update is done automatically by the [Guard] object upon when it falls out of
    /// scope. Since this exclusively borrows the tracker, the scope of the resulting guard should
    /// be minimized as much as possible.
    pub fn get_mut<'t>(&mut self, tracker: &'t Tracker) -> Guard<'_, 't, T> {
        Guard::new(&mut self.obj, tracker)
    }
}

impl<T, D> std::ops::Deref for Tracked<T, D> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.obj
    }
}

/// What to do with a [Tracked] object if it's dropped without being released. The actual handling
/// of the drop logic is done in the policy type's Drop trait.
pub trait DropPolicy {
    fn on_release(&mut self) {}
}

/// Trivial drop policy that does nothing
#[derive(Eq, PartialEq, PartialOrd, Ord, Debug, Default)]
pub struct NoopDropPolicy;

impl DropPolicy for NoopDropPolicy {}

/// Drop policy that asserts if the object has not been properly released
#[derive(Eq, PartialEq, PartialOrd, Ord, Debug)]
pub struct AssertDropPolicy {
    released: bool,
}

impl Default for AssertDropPolicy {
    fn default() -> Self {
        Self { released: false }
    }
}

impl DropPolicy for AssertDropPolicy {
    fn on_release(&mut self) {
        self.released = true;
    }
}

impl Drop for AssertDropPolicy {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            assert!(
                self.released,
                "A memory-tracked value dropped without being released"
            );
        }
    }
}

/// Guard that mutably borrows an object and updates the tracker once modifications are done.
pub struct Guard<'a, 't, T: MemoryUsage> {
    tracker: &'t Tracker,
    object: &'a mut T,
    orig_usage: usize,
}

impl<'a, 't, T: MemoryUsage> Guard<'a, 't, T> {
    fn new(object: &'a mut T, tracker: &'t Tracker) -> Self {
        // Store the original object usage. After all the modifications have been performed,
        // the new size is going to be recorded in the Guard destructor.
        Self {
            tracker,
            orig_usage: object.indirect_memory_usage(),
            object,
        }
    }
}

impl<'a, 't, T: MemoryUsage> Drop for Guard<'a, 't, T> {
    fn drop(&mut self) {
        let cur_usage = self.object.indirect_memory_usage();
        let orig_usage = self.orig_usage;
        match cur_usage.cmp(&orig_usage) {
            std::cmp::Ordering::Equal => (),
            std::cmp::Ordering::Less => self.tracker.sub(orig_usage - cur_usage),
            std::cmp::Ordering::Greater => self.tracker.add(cur_usage - orig_usage),
        }
    }
}

impl<'a, 't, T: MemoryUsage> std::ops::Deref for Guard<'a, 't, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.object
    }
}

impl<'a, 't, T: MemoryUsage> std::ops::DerefMut for Guard<'a, 't, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.object
    }
}

// Code to estimate size taken up by [std::collections::BTreeSet] or [std::collections::BTreeMap].
mod btree {
    // The following structs are laid out in the same way as the real standard library equivalents
    // to give a reasonably precise estimation of their sizes. It is possible that the library
    // implementations change in the future. In that case, the estimation becomes less precise
    // although hopefully will remain good enough for our purposes until the structs below are
    // updated to reflect the change. It's still just an estimate after all.

    const B: usize = 6; // the B parameter for the B-tree
    const BF: usize = 2 * B; // branching factor
    const CAP: usize = BF - 1; // data capacity per node

    struct _LeafNode<K, V> {
        _parent: *mut (),
        _parent_idx: u16,
        _len: u16,
        _keys: [K; CAP],
        _vals: [V; CAP],
    }

    struct _InternalNode<K, V> {
        _data: _LeafNode<K, V>,
        _children: [*mut (); BF],
    }

    /// Estimate the memory usage of the B-tree structure.
    ///
    /// This includes the space taken up by the keys and values stored in the tree. It does NOT
    /// include data pointed to by keys and values indirectly (e.g. via `Box` or `Vec`).
    pub fn usage<K, V>(num_elems: usize) -> usize {
        // Use u64 internally to avoid possible overflow issues on 32-bit platforms
        let num_elems = num_elems as u64;

        // Size of B-tree nodes:
        let leaf_size = std::mem::size_of::<_LeafNode<K, V>>() as u64;
        let internal_size = std::mem::size_of::<_InternalNode<K, V>>() as u64;

        // Size of all leaf elements.
        let leaves = (leaf_size * num_elems) / CAP as u64;

        // Size of internal nodes. We add extra 10% overhead for all the levels of the tree
        let elems_per_internal_node = (CAP * BF) as u64;
        let internals = (internal_size * num_elems * 11) / (elems_per_internal_node * 10);

        // Total size of the B-tree structure. Assuming nodes are on average 75% full,
        // an additional overhead is added for the unused occupied space.
        let total = 4 * (leaves + internals) / 3;

        total as usize
    }
}

/// Trait for data types capable of reporting their current memory usage
///
/// TODO: Make this a derivable trait so the `impl`s react automatically to changes.
pub trait MemoryUsage {
    /// Get amount of memory taken by the data owned by `self` (e.g. if it contains `Box` or `Vec`)
    fn indirect_memory_usage(&self) -> usize;

    fn total_memory_usage(&self) -> usize
    where
        Self: Sized,
    {
        self.indirect_memory_usage() + mem::size_of::<Self>()
    }
}

impl MemoryUsage for u8 {
    fn indirect_memory_usage(&self) -> usize {
        0
    }
}

impl<K, V> MemoryUsage for std::collections::BTreeMap<K, V> {
    /// The mem usage for [BTreeMap].
    ///
    /// Includes the nodes and the key and value data stored in the nodes. It does not, however,
    /// include the memory taken up by data keys and values point to indirectly. Any indirect data
    /// has to be tracked separately. This is so that the memory usage of the B-tree map/set can be
    /// calculated from the number of elements alone without any expensive traversals.
    fn indirect_memory_usage(&self) -> usize {
        btree::usage::<K, V>(self.len())
    }
}

impl<K> MemoryUsage for std::collections::BTreeSet<K> {
    /// Same limitation as for `BTreeMap` also applies here
    fn indirect_memory_usage(&self) -> usize {
        btree::usage::<K, ()>(self.len())
    }
}

impl<T: MemoryUsage> MemoryUsage for Option<T> {
    fn indirect_memory_usage(&self) -> usize {
        self.as_ref().map_or(0, |x| x.indirect_memory_usage())
    }
}

impl<T: MemoryUsage> MemoryUsage for &[T] {
    fn indirect_memory_usage(&self) -> usize {
        self.iter().map(T::indirect_memory_usage).sum::<usize>() + self.len() * mem::size_of::<T>()
    }
}

impl<T: MemoryUsage> MemoryUsage for Vec<T> {
    fn indirect_memory_usage(&self) -> usize {
        self.as_slice().indirect_memory_usage()
    }
}

impl<T: MemoryUsage> MemoryUsage for Box<T> {
    fn indirect_memory_usage(&self) -> usize {
        T::total_memory_usage(self.as_ref())
    }
}

impl MemoryUsage for TxMempoolEntry {
    fn indirect_memory_usage(&self) -> usize {
        self.transaction().indirect_memory_usage()
    }
}

impl MemoryUsage for SignedTransaction {
    /// Only data included indirectly (via pointers). The actual object usage is already contained
    /// in the B-tree map usage.
    fn indirect_memory_usage(&self) -> usize {
        let ins = self.inputs().indirect_memory_usage();
        let outs = self.outputs().indirect_memory_usage();
        let sigs = self.signatures().indirect_memory_usage();
        ins + outs + sigs
    }
}

impl MemoryUsage for TxInput {
    fn indirect_memory_usage(&self) -> usize {
        // No data owned by this object
        0
    }
}

impl MemoryUsage for TxOutput {
    fn indirect_memory_usage(&self) -> usize {
        match self {
            TxOutput::Transfer(_, _) => 0,
            TxOutput::LockThenTransfer(_, _, _) => 0,
            TxOutput::Burn(_) => 0,
            TxOutput::CreateStakePool(_, pd) => pd.indirect_memory_usage(),
            TxOutput::ProduceBlockFromStake(_, _) => 0,
            TxOutput::CreateDelegationId(_, _) => 0,
            TxOutput::DelegateStaking(_, _) => 0,
        }
    }
}

impl MemoryUsage for StakePoolData {
    fn indirect_memory_usage(&self) -> usize {
        0
    }
}

impl MemoryUsage for InputWitness {
    fn indirect_memory_usage(&self) -> usize {
        match self {
            InputWitness::NoSignature(data) => data.indirect_memory_usage(),
            InputWitness::Standard(sig) => sig.raw_signature().indirect_memory_usage(),
        }
    }
}

/// Types where the object created by T::default() takes no indirect memory.
pub trait ZeroUsageDefault: MemoryUsage + Default {}

impl<K, V> ZeroUsageDefault for std::collections::BTreeMap<K, V> {}
impl<K> ZeroUsageDefault for std::collections::BTreeSet<K> {}
