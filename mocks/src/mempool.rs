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

#![allow(clippy::unwrap_used)]

use std::sync::{
    atomic::{AtomicBool, Ordering::Relaxed},
    Arc,
};

use common::{
    chain::{GenBlock, SignedTransaction, Transaction},
    primitives::Id,
};
use mempool::{
    error::{Error, TxValidationError},
    tx_accumulator::TransactionAccumulator,
    MempoolEvent, MempoolInterface, MempoolSubsystemInterface,
};
use subsystem::{subsystem::CallError, CallRequest, ShutdownRequest};

#[derive(Clone)]
pub struct MempoolInterfaceMock {
    pub add_transaction_called: Arc<AtomicBool>,
    pub add_transaction_should_error: Arc<AtomicBool>,
    pub get_all_called: Arc<AtomicBool>,
    pub get_all_should_error: Arc<AtomicBool>,
    pub contains_transaction_called: Arc<AtomicBool>,
    pub contains_transaction_should_error: Arc<AtomicBool>,
    pub collect_txs_called: Arc<AtomicBool>,
    pub collect_txs_should_error: Arc<AtomicBool>,
    pub subscribe_to_events_called: Arc<AtomicBool>,
    pub subscribe_to_events_should_error: Arc<AtomicBool>,
    pub run_called: Arc<AtomicBool>,
    pub run_should_error: Arc<AtomicBool>,
}

impl Default for MempoolInterfaceMock {
    fn default() -> Self {
        Self::new()
    }
}

impl MempoolInterfaceMock {
    pub fn new() -> MempoolInterfaceMock {
        MempoolInterfaceMock {
            add_transaction_called: Arc::new(AtomicBool::new(false)),
            add_transaction_should_error: Arc::new(AtomicBool::new(false)),
            get_all_called: Arc::new(AtomicBool::new(false)),
            get_all_should_error: Arc::new(AtomicBool::new(false)),
            contains_transaction_called: Arc::new(AtomicBool::new(false)),
            contains_transaction_should_error: Arc::new(AtomicBool::new(false)),
            collect_txs_called: Arc::new(AtomicBool::new(false)),
            collect_txs_should_error: Arc::new(AtomicBool::new(false)),
            subscribe_to_events_called: Arc::new(AtomicBool::new(false)),
            subscribe_to_events_should_error: Arc::new(AtomicBool::new(false)),
            run_called: Arc::new(AtomicBool::new(false)),
            run_should_error: Arc::new(AtomicBool::new(false)),
        }
    }
}

const SUBSYSTEM_ERROR: Error =
    Error::Validity(TxValidationError::CallError(CallError::ResultFetchFailed));

#[async_trait::async_trait]
impl MempoolInterface for MempoolInterfaceMock {
    fn add_transaction(&mut self, _tx: SignedTransaction) -> Result<(), Error> {
        self.add_transaction_called.store(true, Relaxed);

        if self.add_transaction_should_error.load(Relaxed) {
            Err(SUBSYSTEM_ERROR)
        } else {
            Ok(())
        }
    }

    fn get_all(&self) -> Result<Vec<SignedTransaction>, Error> {
        self.get_all_called.store(true, Relaxed);

        if self.get_all_should_error.load(Relaxed) {
            Err(SUBSYSTEM_ERROR)
        } else {
            Ok(vec![])
        }
    }

    fn contains_transaction(&self, _tx: &Id<Transaction>) -> Result<bool, Error> {
        self.contains_transaction_called.store(true, Relaxed);

        if self.contains_transaction_should_error.load(Relaxed) {
            Err(SUBSYSTEM_ERROR)
        } else {
            Ok(true)
        }
    }

    fn transaction(&self, _id: &Id<Transaction>) -> Result<Option<SignedTransaction>, Error> {
        unimplemented!()
    }

    fn best_block_id(&self) -> Id<GenBlock> {
        unimplemented!()
    }

    fn collect_txs(
        &self,
        tx_accumulator: Box<dyn TransactionAccumulator + Send>,
    ) -> Result<Box<dyn TransactionAccumulator>, Error> {
        self.collect_txs_called.store(true, Relaxed);

        if self.collect_txs_should_error.load(Relaxed) {
            Err(SUBSYSTEM_ERROR)
        } else {
            Ok(tx_accumulator)
        }
    }

    fn subscribe_to_events(
        &mut self,
        _handler: Arc<dyn Fn(MempoolEvent) + Send + Sync>,
    ) -> Result<(), Error> {
        self.subscribe_to_events_called.store(true, Relaxed);

        if self.subscribe_to_events_should_error.load(Relaxed) {
            Err(SUBSYSTEM_ERROR)
        } else {
            Ok(())
        }
    }
}

#[async_trait::async_trait]
impl MempoolSubsystemInterface for MempoolInterfaceMock {
    async fn run(
        mut self,
        mut call_rq: CallRequest<dyn MempoolInterface>,
        mut shut_rq: ShutdownRequest,
    ) {
        self.run_called.store(true, Relaxed);

        if !self.run_should_error.load(Relaxed) {
            tokio::select! {
                call = call_rq.recv() => call(&mut self).await,
                () = shut_rq.recv() => return,
            }
        }
    }
}
