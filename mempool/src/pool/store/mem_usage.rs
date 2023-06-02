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

mod btree {
    use std::{marker::PhantomData, mem};

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

    impl<K, V> Tree<K, V> {
        pub fn overhead(num_elems: usize) -> usize {
            // Size of B-tree nodes:
            let leaf_size = mem::size_of::<_LeafNode<K, V>>() as u64;
            let internal_size = mem::size_of::<_InternalNode<K, V>>() as u64;

            // Use u64 internally to avoid issues on 64-bit platforms
            let num_elems = num_elems as u64;

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
}
