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

use std::mem;

use common::chain::{
    signature::inputsig::InputWitness, stakelock::StakePoolData, SignedTransaction, TxInput,
    TxOutput,
};

/// Structure that stores the current memory usage and keeps track of its changes
pub struct Tracker {
    current_usage: usize,
}

impl Tracker {
    pub fn new() -> Self {
        Self { current_usage: 0 }
    }

    pub fn get_usage(&self) -> usize {
        self.current_usage
    }
}

/// A data structure which has its memory consumption tracked
#[must_use = "Memory-tracked object dropped without using Tracked::release"]
#[derive(Eq, PartialEq, PartialOrd, Ord)]
pub struct Tracked<T>(T);

impl<T: MemUsage> Tracked<T> {
    /// Create a new object with tracked memory usage
    pub fn new(tracker: &mut Tracker, obj: T) -> Self {
        tracker.current_usage += obj.indirect_memory_usage();
        Self(obj)
    }

    /// Release the object from the tracker and return it as a value
    pub fn release(this: Self, tracker: &mut Tracker) -> T {
        tracker.current_usage -= this.0.indirect_memory_usage();
        this.0
    }

    /// Get mutable access to the tracked object.
    ///
    /// Memory usage update is done automatically by the [Guard] object upon when it falls out of
    /// scope. Since this exclusively borrows the tracker, the scope of the resulting guard should
    /// be minimized as much as possible.
    pub fn get_mut<'t>(&mut self, tracker: &'t mut Tracker) -> Guard<'_, 't, T> {
        Guard::new(&mut self.0, tracker)
    }
}

impl<T> std::ops::Deref for Tracked<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

/// Guard that mutably borrows and object and updates the mem tracker once modifications are done.
pub struct Guard<'a, 't, T: MemUsage> {
    tracker: &'t mut Tracker,
    object: &'a mut T,
}

impl<'a, 't, T: MemUsage> Guard<'a, 't, T> {
    fn new(object: &'a mut T, tracker: &'t mut Tracker) -> Self {
        // Deduct the usage of this object. After the modifications have been done, the new size is
        // going to be added in the Guard destructor. Since the tracker is borrowed mutably for the
        // lifetime of the guard, the tracker cannot be queried while the object is being modified.
        tracker.current_usage -= object.indirect_memory_usage();

        Self { tracker, object }
    }
}

impl<'a, 't, T: MemUsage> Drop for Guard<'a, 't, T> {
    fn drop(&mut self) {
        self.tracker.current_usage += self.object.indirect_memory_usage();
    }
}

impl<'a, 't, T: MemUsage> std::ops::Deref for Guard<'a, 't, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.object
    }
}

impl<'a, 't, T: MemUsage> std::ops::DerefMut for Guard<'a, 't, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.object
    }
}

// Code to estimate size taken up by [std::collections::BTreeSet] or [std::collections::BTreeMap].
mod btree {
    use std::{marker::PhantomData, mem};

    // The following structs are laid out in the same way as the real standard library equivalents
    // to give a reasonably precise estimation of their sizes. It is possible that the library
    // implementations change in the future. In that case, the estimation becomes less precise
    // although hopefully will remain good enough for our purposes until the structs below are
    // updated to reflect the change. It's still just an estimate after all.

    const B: usize = 6;
    const CAP: usize = B * 2 - 1;
    const BF: usize = 2 * B; // branching factor

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

    pub struct Tree<K, V>(PhantomData<fn() -> (K, V)>);

    /// Estimate the memory usage of the B-tree structure.
    ///
    /// This includes the space taken up by the keys and values stored in the tree. It does NOT
    pub fn usage<K, V>(num_elems: usize) -> usize {
        // Use u64 internally to avoid possible overflow issues on 32-bit platforms
        let num_elems = num_elems as u64;

        // Size of B-tree nodes:
        let leaf_size = mem::size_of::<_LeafNode<K, V>>() as u64;
        let internal_size = mem::size_of::<_InternalNode<K, V>>() as u64;

        // Size of all leaf elements.
        let leaves = (leaf_size * num_elems) / CAP as u64;

        // Size of internal nodes. We add extra 10% overhead for all the levels of the tree
        let elems_per_internal_node = (CAP * BF) as u64;
        let internals = (internal_size * num_elems * 11) / (elems_per_internal_node * 10);

        // Total size of the B-tree structure. Assuming nodes are on average 75% full, an
        // additional overhead is added for the unused occupied space.
        let total = 4 * (leaves + internals) / 3;

        total as usize
    }
}

/// Trait for data types capable of reporting their current memory usage
///
/// TODO: Make this a derivable trait so the `impl`s react automatically to changes.
pub trait MemUsage {
    /// Get amount of memory taken by the data owned by `self` (e.g. if it contains `Box` or `Vec`)
    fn indirect_memory_usage(&self) -> usize;

    fn total_memory_usage(&self) -> usize
    where
        Self: Sized,
    {
        self.indirect_memory_usage() + mem::size_of::<Self>()
    }
}

impl MemUsage for u8 {
    fn indirect_memory_usage(&self) -> usize {
        0
    }
}

impl<K, V> MemUsage for std::collections::BTreeMap<K, V> {
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

impl<K> MemUsage for std::collections::BTreeSet<K> {
    /// Same limitation as for `BTreeMap` also applies here
    fn indirect_memory_usage(&self) -> usize {
        btree::usage::<K, ()>(self.len())
    }
}

impl<T: MemUsage> MemUsage for Option<T> {
    fn indirect_memory_usage(&self) -> usize {
        self.as_ref().map_or(0, |x| x.indirect_memory_usage())
    }
}

impl<T: MemUsage> MemUsage for [T] {
    fn indirect_memory_usage(&self) -> usize {
        self.iter().map(T::indirect_memory_usage).sum::<usize>() + self.len() * mem::size_of::<T>()
    }
}

impl<T: MemUsage> MemUsage for Vec<T> {
    fn indirect_memory_usage(&self) -> usize {
        self.as_slice().indirect_memory_usage()
    }
}

impl MemUsage for SignedTransaction {
    /// Only data included indirectly (via pointers). The actual object usage is already contained
    /// in the B-tree map usage.
    fn indirect_memory_usage(&self) -> usize {
        let ins = self.inputs().indirect_memory_usage();
        let outs = self.outputs().indirect_memory_usage();
        let sigs = self.signatures().indirect_memory_usage();
        ins + outs + sigs
    }
}

impl MemUsage for TxInput {
    fn indirect_memory_usage(&self) -> usize {
        // No data owned by this object
        0
    }
}

impl MemUsage for TxOutput {
    fn indirect_memory_usage(&self) -> usize {
        match self {
            TxOutput::Transfer(_, _) => 0,
            TxOutput::LockThenTransfer(_, _, _) => 0,
            TxOutput::Burn(_) => 0,
            TxOutput::CreateStakePool(_, _) => mem::size_of::<StakePoolData>(),
            TxOutput::ProduceBlockFromStake(_, _) => 0,
            TxOutput::CreateDelegationId(_, _) => 0,
            TxOutput::DelegateStaking(_, _) => 0,
        }
    }
}

impl MemUsage for InputWitness {
    fn indirect_memory_usage(&self) -> usize {
        match self {
            InputWitness::NoSignature(data) => data.indirect_memory_usage(),
            InputWitness::Standard(sig) => sig.raw_signature().indirect_memory_usage(),
        }
    }
}
