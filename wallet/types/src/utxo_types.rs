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

use common::chain::TxOutput;

pub type UtxoTypeInt = u16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum UtxoType {
    Transfer = 1 << 0,
    LockThenTransfer = 1 << 1,
    CreateStakePool = 1 << 2,
    Burn = 1 << 3,
    ProduceBlockFromStake = 1 << 4,
    CreateDelegationId = 1 << 5,
    DelegateStaking = 1 << 6,
}

pub fn get_utxo_type(output: &TxOutput) -> UtxoType {
    match output {
        TxOutput::Transfer(_, _) => UtxoType::Transfer,
        TxOutput::LockThenTransfer(_, _, _) => UtxoType::LockThenTransfer,
        TxOutput::Burn(_) => UtxoType::Burn,
        TxOutput::CreateStakePool(_, _) => UtxoType::CreateStakePool,
        TxOutput::ProduceBlockFromStake(_, _) => UtxoType::ProduceBlockFromStake,
        TxOutput::CreateDelegationId(_, _) => UtxoType::CreateDelegationId,
        TxOutput::DelegateStaking(_, _) => UtxoType::DelegateStaking,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UtxoTypes(UtxoTypeInt);

impl UtxoTypes {
    pub const ALL: UtxoTypes = UtxoTypes(UtxoTypeInt::MAX);
}

impl std::ops::BitOr<UtxoType> for UtxoTypes {
    type Output = UtxoTypes;

    fn bitor(self, rhs: UtxoType) -> Self::Output {
        Self(self.0 | rhs as UtxoTypeInt)
    }
}

impl std::ops::BitOr<UtxoType> for UtxoType {
    type Output = UtxoTypes;

    fn bitor(self, rhs: UtxoType) -> Self::Output {
        UtxoTypes::from(self) | rhs
    }
}

impl From<UtxoType> for UtxoTypes {
    fn from(value: UtxoType) -> Self {
        Self(value as UtxoTypeInt)
    }
}

impl UtxoTypes {
    pub fn contains(&self, value: UtxoType) -> bool {
        (self.0 & value as UtxoTypeInt) != 0
    }
}
