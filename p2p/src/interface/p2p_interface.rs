// Copyright (c) 2021-2022 RBB S.r.l
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

use std::sync::Arc;

use common::chain::SignedTransaction;

use crate::{interface::types::ConnectedPeer, types::peer_id::PeerId, P2pEvent};

#[async_trait::async_trait]
pub trait P2pInterface: Send + Sync {
    async fn connect(&mut self, addr: String) -> crate::Result<()>;
    async fn disconnect(&mut self, peer_id: PeerId) -> crate::Result<()>;

    async fn get_peer_count(&self) -> crate::Result<usize>;
    async fn get_bind_addresses(&self) -> crate::Result<Vec<String>>;
    async fn get_connected_peers(&self) -> crate::Result<Vec<ConnectedPeer>>;

    async fn add_reserved_node(&mut self, addr: String) -> crate::Result<()>;
    async fn remove_reserved_node(&mut self, addr: String) -> crate::Result<()>;

    async fn submit_transaction(&mut self, tx: SignedTransaction) -> crate::Result<()>;

    fn subscribe_to_events(
        &mut self,
        handler: Arc<dyn Fn(P2pEvent) + Send + Sync>,
    ) -> crate::Result<()>;
}
