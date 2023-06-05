// Copyright (c) 2023 RBB S.r.l
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

use chainstate::ConnectTransactionError;

/// Check an error signifies a potential orphan transaction
pub fn is_orphan_error(err: &ConnectTransactionError) -> bool {
    use ConnectTransactionError as CTE;
    match err {
        // This signifies a possible orphan
        CTE::MissingOutputOrSpent => true,
        // These do not
        CTE::StorageError(_)
        | CTE::TxNumWrongInBlockOnConnect(_, _)
        | CTE::TxNumWrongInBlockOnDisconnect(_, _)
        | CTE::InvariantBrokenAlreadyUnspent
        | CTE::MissingTxInputs
        | CTE::MissingTxUndo(_)
        | CTE::MissingBlockUndo(_)
        | CTE::MissingBlockRewardUndo(_)
        | CTE::MissingMempoolTxsUndo
        | CTE::TxUndoWithDependency(_)
        | CTE::AttemptToPrintMoney(_, _)
        | CTE::BlockRewardInputOutputMismatch(_, _)
        | CTE::TxFeeTotalCalcFailed(_, _)
        | CTE::SignatureVerificationFailed(_)
        | CTE::BlockHeightArithmeticError
        | CTE::BlockTimestampArithmeticError
        | CTE::InvariantErrorHeaderCouldNotBeLoaded(_)
        | CTE::InvariantErrorHeaderCouldNotBeLoadedFromHeight(_, _)
        | CTE::BlockIndexCouldNotBeLoaded(_)
        | CTE::FailedToAddAllFeesOfBlock(_)
        | CTE::RewardAdditionError(_)
        | CTE::TimeLockViolation(_)
        | CTE::UtxoError(_)
        | CTE::TokensError(_)
        | CTE::TxIndexError(_)
        | CTE::TransactionVerifierError(_)
        | CTE::UtxoBlockUndoError(_)
        | CTE::AccountingBlockUndoError(_)
        | CTE::BurnAmountSumError(_)
        | CTE::AttemptToSpendBurnedAmount
        | CTE::PoSAccountingError(_)
        | CTE::MissingPoSAccountingUndo(_)
        | CTE::SpendStakeError(_)
        | CTE::InvalidInputTypeInTx
        | CTE::InvalidOutputTypeInTx
        | CTE::InvalidInputTypeInReward
        | CTE::InvalidOutputTypeInReward
        | CTE::PoolOwnerBalanceNotFound(_)
        | CTE::PoolDataNotFound(_)
        | CTE::PoolOwnerRewardCalculationFailed(_, _)
        | CTE::PoolOwnerRewardCannotExceedTotalReward(..)
        | CTE::DelegationsRewardSumFailed(..)
        | CTE::DelegationRewardOverflow(..)
        | CTE::DistributedDelegationsRewardExceedTotal(..)
        | CTE::TotalDelegationBalanceZero(_)
        | CTE::UndoFetchFailure
        | CTE::TxVerifierStorage
        | CTE::DestinationRetrievalError(_)
        | CTE::DelegationDataNotFound(_)
        | CTE::OutputTimelockError(_)
        | CTE::NotEnoughPledgeToCreateStakePool(..) => false,
    }
}
