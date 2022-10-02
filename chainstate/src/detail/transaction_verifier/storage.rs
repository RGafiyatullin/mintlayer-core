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

use chainstate_types::{storage_result, GenBlockIndex};
use common::{
    chain::{
        tokens::{TokenAuxiliaryData, TokenId},
        GenBlock, OutPointSourceId, Transaction, TxMainChainIndex,
    },
    primitives::{BlockHeight, Id},
};
use thiserror::Error;
use utxo::{FlushableUtxoView, UtxosStorageRead, UtxosUndoStorageWrite};

use super::error::{TokensError, TxIndexError};

#[derive(Error, Debug, PartialEq, Eq, Clone)]
pub enum TransactionVerifierStorageError {
    #[error("Gen block index not found")]
    GenBlockIndexRetrievalFailed(Id<GenBlock>),
    #[error("Failed to persist state: {0}")]
    StatePersistenceError(#[from] chainstate_types::StatePersistenceError),
    #[error("Failed to persist state: {0}")]
    GetAncestorError(#[from] chainstate_types::GetAncestorError),
    #[error("Tokens error: {0}")]
    TokensError(#[from] TokensError),
    #[error("Utxo error: {0}")]
    UtxoError(#[from] utxo::Error),
    #[error("Storage error: {0}")]
    StorageError(#[from] storage_result::Error),
    #[error("Tx index error: {0}")]
    TxIndexError(#[from] TxIndexError),
}

pub trait TransactionVerifierStorageRef: UtxosStorageRead {
    fn get_token_id_from_issuance_tx(
        &self,
        tx_id: Id<Transaction>,
    ) -> Result<Option<TokenId>, TransactionVerifierStorageError>;

    fn get_gen_block_index(
        &self,
        block_id: &Id<GenBlock>,
    ) -> Result<Option<GenBlockIndex>, storage_result::Error>;

    fn get_ancestor(
        &self,
        block_index: &GenBlockIndex,
        target_height: BlockHeight,
    ) -> Result<GenBlockIndex, TransactionVerifierStorageError>;

    fn get_mainchain_tx_index(
        &self,
        tx_id: &OutPointSourceId,
    ) -> Result<Option<TxMainChainIndex>, TransactionVerifierStorageError>;

    fn get_token_aux_data(
        &self,
        token_id: &TokenId,
    ) -> Result<Option<TokenAuxiliaryData>, TransactionVerifierStorageError>;
}

pub trait TransactionVerifierStorageMut:
    TransactionVerifierStorageRef + UtxosUndoStorageWrite + FlushableUtxoView
{
    fn set_mainchain_tx_index(
        &mut self,
        tx_id: &OutPointSourceId,
        tx_index: &TxMainChainIndex,
    ) -> Result<(), TransactionVerifierStorageError>;

    fn del_mainchain_tx_index(
        &mut self,
        tx_id: &OutPointSourceId,
    ) -> Result<(), TransactionVerifierStorageError>;

    fn set_token_aux_data(
        &mut self,
        token_id: &TokenId,
        data: &TokenAuxiliaryData,
    ) -> Result<(), TransactionVerifierStorageError>;

    fn del_token_aux_data(
        &mut self,
        token_id: &TokenId,
    ) -> Result<(), TransactionVerifierStorageError>;

    fn set_token_id(
        &mut self,
        issuance_tx_id: &Id<Transaction>,
        token_id: &TokenId,
    ) -> Result<(), TransactionVerifierStorageError>;

    fn del_token_id(
        &mut self,
        issuance_tx_id: &Id<Transaction>,
    ) -> Result<(), TransactionVerifierStorageError>;
}
