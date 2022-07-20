use std::sync::Arc;

use chainstate_storage::{BlockchainStorageRead, Store};
use common::{
    chain::{
        block::{timestamp::BlockTimestamp, Block},
        config::create_unit_test_config,
        signature::inputsig::InputWitness,
        timelock::OutputTimeLock,
        OutPointSourceId, OutputPurpose, Transaction, TxInput, TxOutput,
    },
    primitives::{time, Amount, BlockHeight, Idable},
};

use crate::{
    detail::{
        spend_cache::error::StateUpdateError,
        tests::{
            anyonecanspend_address, ERR_BEST_BLOCK_NOT_FOUND, ERR_CREATE_BLOCK_FAIL,
            ERR_CREATE_TX_FAIL,
        },
    },
    BlockError, BlockSource, Chainstate, ChainstateConfig,
};

#[test]
fn output_lock_until_height() {
    common::concurrency::model(|| {
        let chain_config = Arc::new(create_unit_test_config());
        let chainstate_config = ChainstateConfig::new();
        let storage = Store::new_empty().unwrap();
        let mut chainstate = Chainstate::new_no_genesis(
            chain_config,
            chainstate_config,
            storage,
            None,
            Default::default(),
        )
        .unwrap();

        let block_height_that_unlocks = 10;

        // Process the genesis block.
        chainstate
            .process_block(
                chainstate.chain_config.genesis_block().clone(),
                BlockSource::Local,
            )
            .unwrap();
        assert_eq!(
            chainstate
                .chainstate_storage
                .get_best_block_id()
                .expect(ERR_BEST_BLOCK_NOT_FOUND),
            Some(chainstate.chain_config.genesis_block_id())
        );

        // create the first block, with a locked output
        {
            let prev_block = chainstate.chain_config.genesis_block();

            let outputs = vec![
                TxOutput::new(
                    Amount::from_atoms(100000),
                    OutputPurpose::Transfer(anyonecanspend_address()),
                ),
                TxOutput::new(
                    Amount::from_atoms(100000),
                    OutputPurpose::LockThenTransfer(
                        anyonecanspend_address(),
                        OutputTimeLock::UntilHeight(BlockHeight::new(block_height_that_unlocks)),
                    ),
                ),
            ];

            let inputs = vec![TxInput::new(
                OutPointSourceId::Transaction(prev_block.transactions().get(0).unwrap().get_id()),
                0,
                InputWitness::NoSignature(None),
            )];

            let block = Block::new(
                vec![Transaction::new(0, inputs, outputs, 0).expect(ERR_CREATE_TX_FAIL)],
                Some(prev_block.get_id()),
                BlockTimestamp::from_duration_since_epoch(time::get()).unwrap(),
                common::chain::block::ConsensusData::None,
            )
            .expect(ERR_CREATE_BLOCK_FAIL);
            chainstate.process_block(block, BlockSource::Local).unwrap();

            assert_eq!(
                chainstate.get_best_block_index().unwrap().unwrap().block_height(),
                BlockHeight::new(1)
            );
        }

        let locked_output = {
            let prev_block_id =
                chainstate.get_block_id_from_height(&BlockHeight::new(1)).unwrap().unwrap();
            let prev_block = chainstate.get_block(prev_block_id).unwrap().unwrap();
            TxInput::new(
                OutPointSourceId::Transaction(prev_block.transactions().get(0).unwrap().get_id()),
                1,
                InputWitness::NoSignature(None),
            )
        };

        // attempt to create the next block, and attempt to spend the locked output
        {
            let prev_block_id =
                chainstate.get_best_block_index().unwrap().unwrap().block_id().clone();
            let prev_block = chainstate.get_block(prev_block_id).unwrap().unwrap();

            let outputs = vec![TxOutput::new(
                Amount::from_atoms(5000),
                OutputPurpose::Transfer(anyonecanspend_address()),
            )];

            let inputs = vec![locked_output.clone()];

            let block = Block::new(
                vec![Transaction::new(0, inputs, outputs, 0).expect(ERR_CREATE_TX_FAIL)],
                Some(prev_block.get_id()),
                BlockTimestamp::from_duration_since_epoch(time::get()).unwrap(),
                common::chain::block::ConsensusData::None,
            )
            .expect(ERR_CREATE_BLOCK_FAIL);
            assert_eq!(
                chainstate.process_block(block, BlockSource::Local).unwrap_err(),
                BlockError::StateUpdateFailed(StateUpdateError::TimeLockViolation)
            );

            assert_eq!(
                chainstate.get_best_block_index().unwrap().unwrap().block_height(),
                BlockHeight::new(1)
            );
        }

        // create another block, and spend the first input from the previous block
        {
            let prev_block_id =
                chainstate.get_block_id_from_height(&BlockHeight::new(1)).unwrap().unwrap();
            let prev_block = chainstate.get_block(prev_block_id).unwrap().unwrap();

            let outputs = vec![TxOutput::new(
                Amount::from_atoms(10000),
                OutputPurpose::Transfer(anyonecanspend_address()),
            )];

            let inputs = vec![TxInput::new(
                OutPointSourceId::Transaction(prev_block.transactions().get(0).unwrap().get_id()),
                0,
                InputWitness::NoSignature(None),
            )];

            let block = Block::new(
                vec![Transaction::new(0, inputs, outputs, 0).expect(ERR_CREATE_TX_FAIL)],
                Some(prev_block.get_id()),
                BlockTimestamp::from_duration_since_epoch(time::get()).unwrap(),
                common::chain::block::ConsensusData::None,
            )
            .expect(ERR_CREATE_BLOCK_FAIL);
            chainstate.process_block(block, BlockSource::Local).unwrap();

            assert_eq!(
                chainstate.get_best_block_index().unwrap().unwrap().block_height(),
                BlockHeight::new(2)
            );
        }

        // let's create more blocks until block_height_that_unlocks - 1, and always fail to spend, and build up the chain
        for height in 3..block_height_that_unlocks {
            assert_eq!(
                BlockHeight::new(height - 1),
                chainstate.get_best_block_index().unwrap().unwrap().block_height()
            );
            logging::log::info!("Submitting block of height: {}", height);
            // create another block, and spend the first input from the previous block
            {
                let prev_block_id =
                    chainstate.get_best_block_index().unwrap().unwrap().block_id().clone();
                let prev_block = chainstate.get_block(prev_block_id).unwrap().unwrap();

                let outputs = vec![TxOutput::new(
                    Amount::from_atoms(5000),
                    OutputPurpose::Transfer(anyonecanspend_address()),
                )];

                let inputs = vec![locked_output.clone()];

                let block = Block::new(
                    vec![Transaction::new(0, inputs, outputs, 0).expect(ERR_CREATE_TX_FAIL)],
                    Some(prev_block.get_id()),
                    BlockTimestamp::from_duration_since_epoch(time::get()).unwrap(),
                    common::chain::block::ConsensusData::None,
                )
                .expect(ERR_CREATE_BLOCK_FAIL);
                assert_eq!(
                    chainstate.process_block(block.clone(), BlockSource::Local).unwrap_err(),
                    BlockError::StateUpdateFailed(StateUpdateError::TimeLockViolation)
                );

                assert_eq!(
                    chainstate.get_best_block_index().unwrap().unwrap().block_height(),
                    BlockHeight::new(height - 1)
                );
            }

            // create another block, with no transactions, and get the blockchain to progress
            {
                let prev_block_id =
                    chainstate.get_best_block_index().unwrap().unwrap().block_id().clone();
                let prev_block = chainstate.get_block(prev_block_id).unwrap().unwrap();

                let block = Block::new(
                    vec![],
                    Some(prev_block.get_id()),
                    BlockTimestamp::from_duration_since_epoch(time::get()).unwrap(),
                    common::chain::block::ConsensusData::None,
                )
                .expect(ERR_CREATE_BLOCK_FAIL);
                chainstate.process_block(block.clone(), BlockSource::Local).unwrap();

                assert_eq!(
                    chainstate.get_best_block_index().unwrap().unwrap().block_height(),
                    BlockHeight::new(height)
                );
            }
        }

        // now we should be able to spend it at block_height_that_unlocks
        {
            let height = block_height_that_unlocks;
            let prev_block_id = chainstate
                .get_block_id_from_height(&BlockHeight::new(height - 1))
                .unwrap()
                .unwrap();
            let prev_block = chainstate.get_block(prev_block_id.clone()).unwrap().unwrap();

            let tip_id = chainstate.get_best_block_index().unwrap().unwrap().block_id().clone();
            assert_eq!(tip_id, prev_block_id);

            let outputs = vec![TxOutput::new(
                Amount::from_atoms(5000),
                OutputPurpose::Transfer(anyonecanspend_address()),
            )];

            let inputs = vec![locked_output];

            let block = Block::new(
                vec![Transaction::new(0, inputs, outputs, 0).expect(ERR_CREATE_TX_FAIL)],
                Some(prev_block.get_id()),
                BlockTimestamp::from_duration_since_epoch(time::get()).unwrap(),
                common::chain::block::ConsensusData::None,
            )
            .expect(ERR_CREATE_BLOCK_FAIL);

            chainstate.process_block(block, BlockSource::Local).unwrap();

            assert_eq!(
                chainstate.get_best_block_index().unwrap().unwrap().block_height(),
                BlockHeight::new(height)
            );
        }
    });
}

#[test]
fn output_lock_until_height_but_spend_at_same_block() {
    common::concurrency::model(|| {
        let chain_config = Arc::new(create_unit_test_config());
        let chainstate_config = ChainstateConfig::new();
        let storage = Store::new_empty().unwrap();
        let mut chainstate = Chainstate::new_no_genesis(
            chain_config,
            chainstate_config,
            storage,
            None,
            Default::default(),
        )
        .unwrap();

        let block_height_that_unlocks = 10;

        // Process the genesis block.
        chainstate
            .process_block(
                chainstate.chain_config.genesis_block().clone(),
                BlockSource::Local,
            )
            .unwrap();
        assert_eq!(
            chainstate
                .chainstate_storage
                .get_best_block_id()
                .expect(ERR_BEST_BLOCK_NOT_FOUND),
            Some(chainstate.chain_config.genesis_block_id())
        );

        // create the first block, with a locked output
        {
            let prev_block = chainstate.chain_config.genesis_block();

            let outputs1 = vec![
                TxOutput::new(
                    Amount::from_atoms(100000),
                    OutputPurpose::Transfer(anyonecanspend_address()),
                ),
                TxOutput::new(
                    Amount::from_atoms(100000),
                    OutputPurpose::LockThenTransfer(
                        anyonecanspend_address(),
                        OutputTimeLock::UntilHeight(BlockHeight::new(block_height_that_unlocks)),
                    ),
                ),
            ];
            let inputs1 = vec![TxInput::new(
                OutPointSourceId::Transaction(prev_block.transactions().get(0).unwrap().get_id()),
                0,
                InputWitness::NoSignature(None),
            )];
            let tx1 = Transaction::new(0, inputs1, outputs1, 0).expect(ERR_CREATE_TX_FAIL);

            let outputs2 = vec![TxOutput::new(
                Amount::from_atoms(50000),
                OutputPurpose::Transfer(anyonecanspend_address()),
            )];
            let inputs2 = vec![TxInput::new(
                OutPointSourceId::Transaction(tx1.get_id()),
                1,
                InputWitness::NoSignature(None),
            )];
            let tx2 = Transaction::new(0, inputs2, outputs2, 0).expect(ERR_CREATE_TX_FAIL);

            let block = Block::new(
                vec![tx1, tx2],
                Some(prev_block.get_id()),
                BlockTimestamp::from_duration_since_epoch(time::get()).unwrap(),
                common::chain::block::ConsensusData::None,
            )
            .expect(ERR_CREATE_BLOCK_FAIL);
            assert_eq!(
                chainstate.process_block(block, BlockSource::Local).unwrap_err(),
                BlockError::StateUpdateFailed(StateUpdateError::TimeLockViolation)
            );

            assert_eq!(
                chainstate.get_best_block_index().unwrap().unwrap().block_height(),
                BlockHeight::new(0)
            );
        }
    });
}

#[test]
fn output_lock_for_block_count() {
    common::concurrency::model(|| {
        let chain_config = Arc::new(create_unit_test_config());
        let chainstate_config = ChainstateConfig::new();
        let storage = Store::new_empty().unwrap();
        let mut chainstate = Chainstate::new_no_genesis(
            chain_config,
            chainstate_config,
            storage,
            None,
            Default::default(),
        )
        .unwrap();

        let block_count_that_unlocks = 20;
        let block_height_with_locked_output = 1;

        // Process the genesis block.
        chainstate
            .process_block(
                chainstate.chain_config.genesis_block().clone(),
                BlockSource::Local,
            )
            .unwrap();
        assert_eq!(
            chainstate
                .chainstate_storage
                .get_best_block_id()
                .expect(ERR_BEST_BLOCK_NOT_FOUND),
            Some(chainstate.chain_config.genesis_block_id())
        );

        // create the first block, with a locked output
        {
            let prev_block = chainstate.chain_config.genesis_block();

            let outputs = vec![
                TxOutput::new(
                    Amount::from_atoms(100000),
                    OutputPurpose::Transfer(anyonecanspend_address()),
                ),
                TxOutput::new(
                    Amount::from_atoms(100000),
                    OutputPurpose::LockThenTransfer(
                        anyonecanspend_address(),
                        OutputTimeLock::ForBlockCount(block_count_that_unlocks),
                    ),
                ),
            ];

            let inputs = vec![TxInput::new(
                OutPointSourceId::Transaction(prev_block.transactions().get(0).unwrap().get_id()),
                0,
                InputWitness::NoSignature(None),
            )];

            let block = Block::new(
                vec![Transaction::new(0, inputs, outputs, 0).expect(ERR_CREATE_TX_FAIL)],
                Some(prev_block.get_id()),
                BlockTimestamp::from_duration_since_epoch(time::get()).unwrap(),
                common::chain::block::ConsensusData::None,
            )
            .expect(ERR_CREATE_BLOCK_FAIL);
            chainstate.process_block(block, BlockSource::Local).unwrap();

            assert_eq!(
                chainstate.get_best_block_index().unwrap().unwrap().block_height(),
                BlockHeight::new(block_height_with_locked_output)
            );
        }

        let locked_output = {
            let prev_block_id =
                chainstate.get_block_id_from_height(&BlockHeight::new(1)).unwrap().unwrap();
            let prev_block = chainstate.get_block(prev_block_id).unwrap().unwrap();
            TxInput::new(
                OutPointSourceId::Transaction(prev_block.transactions().get(0).unwrap().get_id()),
                1,
                InputWitness::NoSignature(None),
            )
        };

        // attempt to create the next block, and attempt to spend the locked output
        {
            let prev_block_id =
                chainstate.get_best_block_index().unwrap().unwrap().block_id().clone();
            let prev_block = chainstate.get_block(prev_block_id).unwrap().unwrap();

            let outputs = vec![TxOutput::new(
                Amount::from_atoms(5000),
                OutputPurpose::Transfer(anyonecanspend_address()),
            )];

            let inputs = vec![locked_output.clone()];

            let block = Block::new(
                vec![Transaction::new(0, inputs, outputs, 0).expect(ERR_CREATE_TX_FAIL)],
                Some(prev_block.get_id()),
                BlockTimestamp::from_duration_since_epoch(time::get()).unwrap(),
                common::chain::block::ConsensusData::None,
            )
            .expect(ERR_CREATE_BLOCK_FAIL);
            assert_eq!(
                chainstate.process_block(block, BlockSource::Local).unwrap_err(),
                BlockError::StateUpdateFailed(StateUpdateError::TimeLockViolation)
            );

            assert_eq!(
                chainstate.get_best_block_index().unwrap().unwrap().block_height(),
                BlockHeight::new(1)
            );
        }

        // create another block, and spend the first input from the previous block
        {
            let prev_block_id =
                chainstate.get_best_block_index().unwrap().unwrap().block_id().clone();
            let prev_block = chainstate.get_block(prev_block_id).unwrap().unwrap();

            let outputs = vec![TxOutput::new(
                Amount::from_atoms(10000),
                OutputPurpose::Transfer(anyonecanspend_address()),
            )];

            let inputs = vec![TxInput::new(
                OutPointSourceId::Transaction(prev_block.transactions().get(0).unwrap().get_id()),
                0,
                InputWitness::NoSignature(None),
            )];

            let block = Block::new(
                vec![Transaction::new(0, inputs, outputs, 0).expect(ERR_CREATE_TX_FAIL)],
                Some(prev_block.get_id()),
                BlockTimestamp::from_duration_since_epoch(time::get()).unwrap(),
                common::chain::block::ConsensusData::None,
            )
            .expect(ERR_CREATE_BLOCK_FAIL);
            chainstate.process_block(block, BlockSource::Local).unwrap();

            assert_eq!(
                chainstate.get_best_block_index().unwrap().unwrap().block_height(),
                BlockHeight::new(2)
            );
        }

        // let's create more blocks until block_count_that_unlocks + block_height_with_locked_output, and always fail to spend, and build up the chain
        for height in 3..block_count_that_unlocks + block_height_with_locked_output {
            assert_eq!(
                BlockHeight::new(height - 1),
                chainstate.get_best_block_index().unwrap().unwrap().block_height()
            );
            logging::log::info!("Submitting block of height: {}", height);
            // create another block, and spend the first input from the previous block
            {
                let prev_block_id =
                    chainstate.get_best_block_index().unwrap().unwrap().block_id().clone();
                let prev_block = chainstate.get_block(prev_block_id).unwrap().unwrap();

                let outputs = vec![TxOutput::new(
                    Amount::from_atoms(5000),
                    OutputPurpose::Transfer(anyonecanspend_address()),
                )];

                let inputs = vec![locked_output.clone()];

                let block = Block::new(
                    vec![Transaction::new(0, inputs, outputs, 0).expect(ERR_CREATE_TX_FAIL)],
                    Some(prev_block.get_id()),
                    BlockTimestamp::from_duration_since_epoch(time::get()).unwrap(),
                    common::chain::block::ConsensusData::None,
                )
                .expect(ERR_CREATE_BLOCK_FAIL);
                assert_eq!(
                    chainstate.process_block(block.clone(), BlockSource::Local).unwrap_err(),
                    BlockError::StateUpdateFailed(StateUpdateError::TimeLockViolation)
                );

                assert_eq!(
                    chainstate.get_best_block_index().unwrap().unwrap().block_height(),
                    BlockHeight::new(height - 1)
                );
            }

            // create another block, with no transactions, and get the blockchain to progress
            {
                let prev_block_id =
                    chainstate.get_best_block_index().unwrap().unwrap().block_id().clone();
                let prev_block = chainstate.get_block(prev_block_id).unwrap().unwrap();

                let block = Block::new(
                    vec![],
                    Some(prev_block.get_id()),
                    BlockTimestamp::from_duration_since_epoch(time::get()).unwrap(),
                    common::chain::block::ConsensusData::None,
                )
                .expect(ERR_CREATE_BLOCK_FAIL);
                chainstate.process_block(block.clone(), BlockSource::Local).unwrap();

                assert_eq!(
                    chainstate.get_best_block_index().unwrap().unwrap().block_height(),
                    BlockHeight::new(height)
                );
            }
        }

        // now we should be able to spend it at block_count_that_unlocks
        {
            let height = block_count_that_unlocks + block_height_with_locked_output;
            let prev_block_id = chainstate
                .get_block_id_from_height(&BlockHeight::new(height - 1))
                .unwrap()
                .unwrap();
            let prev_block = chainstate.get_block(prev_block_id).unwrap().unwrap();

            let outputs = vec![TxOutput::new(
                Amount::from_atoms(5000),
                OutputPurpose::Transfer(anyonecanspend_address()),
            )];

            let inputs = vec![locked_output];

            let block = Block::new(
                vec![Transaction::new(0, inputs, outputs, 0).expect(ERR_CREATE_TX_FAIL)],
                Some(prev_block.get_id()),
                BlockTimestamp::from_duration_since_epoch(time::get()).unwrap(),
                common::chain::block::ConsensusData::None,
            )
            .expect(ERR_CREATE_BLOCK_FAIL);

            chainstate.process_block(block, BlockSource::Local).unwrap();

            assert_eq!(
                chainstate.get_best_block_index().unwrap().unwrap().block_height(),
                BlockHeight::new(height)
            );
        }
    });
}

#[test]
fn output_lock_for_lock_count_but_spend_at_same_block() {
    common::concurrency::model(|| {
        let chain_config = Arc::new(create_unit_test_config());
        let chainstate_config = ChainstateConfig::new();
        let storage = Store::new_empty().unwrap();
        let mut chainstate = Chainstate::new_no_genesis(
            chain_config,
            chainstate_config,
            storage,
            None,
            Default::default(),
        )
        .unwrap();

        let block_count_that_unlocks = 10;

        // Process the genesis block.
        chainstate
            .process_block(
                chainstate.chain_config.genesis_block().clone(),
                BlockSource::Local,
            )
            .unwrap();
        assert_eq!(
            chainstate
                .chainstate_storage
                .get_best_block_id()
                .expect(ERR_BEST_BLOCK_NOT_FOUND),
            Some(chainstate.chain_config.genesis_block_id())
        );

        // create the first block, with a locked output
        {
            let prev_block = chainstate.chain_config.genesis_block();

            let outputs1 = vec![
                TxOutput::new(
                    Amount::from_atoms(100000),
                    OutputPurpose::Transfer(anyonecanspend_address()),
                ),
                TxOutput::new(
                    Amount::from_atoms(100000),
                    OutputPurpose::LockThenTransfer(
                        anyonecanspend_address(),
                        OutputTimeLock::ForBlockCount(block_count_that_unlocks),
                    ),
                ),
            ];
            let inputs1 = vec![TxInput::new(
                OutPointSourceId::Transaction(prev_block.transactions().get(0).unwrap().get_id()),
                0,
                InputWitness::NoSignature(None),
            )];
            let tx1 = Transaction::new(0, inputs1, outputs1, 0).expect(ERR_CREATE_TX_FAIL);

            let outputs2 = vec![TxOutput::new(
                Amount::from_atoms(50000),
                OutputPurpose::Transfer(anyonecanspend_address()),
            )];
            let inputs2 = vec![TxInput::new(
                OutPointSourceId::Transaction(tx1.get_id()),
                1,
                InputWitness::NoSignature(None),
            )];
            let tx2 = Transaction::new(0, inputs2, outputs2, 0).expect(ERR_CREATE_TX_FAIL);

            let block = Block::new(
                vec![tx1, tx2],
                Some(prev_block.get_id()),
                BlockTimestamp::from_duration_since_epoch(time::get()).unwrap(),
                common::chain::block::ConsensusData::None,
            )
            .expect(ERR_CREATE_BLOCK_FAIL);
            assert_eq!(
                chainstate.process_block(block, BlockSource::Local).unwrap_err(),
                BlockError::StateUpdateFailed(StateUpdateError::TimeLockViolation)
            );

            assert_eq!(
                chainstate.get_best_block_index().unwrap().unwrap().block_height(),
                BlockHeight::new(0)
            );
        }
    });
}
