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

use std::collections::BTreeMap;

use common::address::Address;
use crypto::{kdf::KdfChallenge, key::extended::ExtendedPublicKey, symkey::SymmetricKey};
use serialization::{Codec, DecodeAll, Encode, EncodeLike};
use storage::schema;
use utils::maybe_encrypted::{MaybeEncrypted, MaybeEncryptedError};
use wallet_types::{
    AccountDerivationPathId, AccountId, AccountInfo, AccountKeyPurposeId, AccountWalletTxId,
    KeychainUsageState, RootKeyContent, RootKeyId, WalletTx,
};

use crate::{
    schema::{self as db, Schema},
    WalletStorageEncryptionRead, WalletStorageEncryptionWrite, WalletStorageReadLocked,
    WalletStorageReadUnlocked, WalletStorageWriteLocked, WalletStorageWriteUnlocked,
};

mod well_known {
    use crypto::kdf::KdfChallenge;

    use super::Codec;

    /// Pre-defined database keys
    pub trait Entry {
        /// Key for this entry
        const KEY: &'static [u8];
        /// Value type for this entry
        type Value: Codec;
    }

    macro_rules! declare_entry {
        ($name:ident: $type:ty) => {
            pub struct $name;
            impl Entry for $name {
                const KEY: &'static [u8] = stringify!($name).as_bytes();
                type Value = $type;
            }
        };
    }

    declare_entry!(StoreVersion: u32);
    declare_entry!(EncryptionKeyKdfChallenge: KdfChallenge);
}

#[derive(PartialEq, Clone)]
pub enum EncryptionState {
    // The secret parts of the DB (the private keys) are encrypted and we don't have the key to decrypt them
    Locked,
    // If Key is Some then DB is encrypted but we have the key to decrypt it
    // if Key is None then DB is not encrypted
    Unlocked(Option<SymmetricKey>),
}

/// Read-only chainstate storage transaction
pub struct StoreTxRo<'st, B: storage::Backend> {
    storage: storage::TransactionRo<'st, B, Schema>,
}

/// Read-only chainstate storage transaction unlocked
pub struct StoreTxRoUnlocked<'st, B: storage::Backend> {
    storage: storage::TransactionRo<'st, B, Schema>,
    encryption_key: &'st Option<SymmetricKey>,
}

/// Read-write chainstate storage transaction
pub struct StoreTxRw<'st, B: storage::Backend> {
    storage: storage::TransactionRw<'st, B, Schema>,
}

/// Read-write chainstate storage transaction unlocked
pub struct StoreTxRwUnlocked<'st, B: storage::Backend> {
    storage: storage::TransactionRw<'st, B, Schema>,
    encryption_key: &'st Option<SymmetricKey>,
}

impl<'st, B: storage::Backend> StoreTxRo<'st, B> {
    pub fn new(storage: storage::TransactionRo<'st, B, Schema>) -> Self {
        Self { storage }
    }
}

impl<'st, B: storage::Backend> StoreTxRoUnlocked<'st, B> {
    pub fn new(
        storage: storage::TransactionRo<'st, B, Schema>,
        encryption_key: &'st Option<SymmetricKey>,
    ) -> Self {
        Self {
            storage,
            encryption_key,
        }
    }
}

impl<'st, B: storage::Backend> StoreTxRw<'st, B> {
    pub fn new(storage: storage::TransactionRw<'st, B, Schema>) -> Self {
        Self { storage }
    }
}

impl<'st, B: storage::Backend> StoreTxRwUnlocked<'st, B> {
    pub fn new(
        storage: storage::TransactionRw<'st, B, Schema>,
        encryption_key: &'st Option<SymmetricKey>,
    ) -> Self {
        Self {
            storage,
            encryption_key,
        }
    }
}

macro_rules! impl_read_ops {
    ($TxType:ident) => {
        /// Wallet data storage transaction
        impl<'st, B: storage::Backend> WalletStorageReadLocked for $TxType<'st, B> {
            fn get_storage_version(&self) -> crate::Result<u32> {
                self.read_value::<well_known::StoreVersion>().map(|v| v.unwrap_or_default())
            }

            fn get_transaction(&self, id: &AccountWalletTxId) -> crate::Result<Option<WalletTx>> {
                self.read::<db::DBTxs, _, _>(id)
            }

            fn get_accounts_info(&self) -> crate::Result<BTreeMap<AccountId, AccountInfo>> {
                Ok(self.storage.get::<db::DBAccounts, _>().prefix_iter_decoded(&())?.collect())
            }

            fn get_address(&self, id: &AccountDerivationPathId) -> crate::Result<Option<Address>> {
                self.read::<db::DBAddresses, _, _>(id)
            }

            fn get_addresses(
                &self,
                account_id: &AccountId,
            ) -> crate::Result<BTreeMap<AccountDerivationPathId, Address>> {
                self.storage
                    .get::<db::DBAddresses, _>()
                    .prefix_iter_decoded(account_id)
                    .map_err(crate::Error::from)
                    .map(Iterator::collect)
            }

            fn exactly_one_root_key(&self) -> crate::Result<bool> {
                self.storage
                    .get::<db::DBRootKeys, _>()
                    .prefix_iter_decoded(&())
                    .map_err(crate::Error::from)
                    .map(Iterator::count)
                    .map(|count| count == 1)
            }

            /// Collect and return all transactions from the storage
            fn get_transactions(
                &self,
                account_id: &AccountId,
            ) -> crate::Result<BTreeMap<AccountWalletTxId, WalletTx>> {
                self.storage
                    .get::<db::DBTxs, _>()
                    .prefix_iter_decoded(account_id)
                    .map_err(crate::Error::from)
                    .map(Iterator::collect)
            }

            fn get_keychain_usage_state(
                &self,
                id: &AccountKeyPurposeId,
            ) -> crate::Result<Option<KeychainUsageState>> {
                self.read::<db::DBKeychainUsageStates, _, _>(id)
            }

            fn get_keychain_usage_states(
                &self,
                account_id: &AccountId,
            ) -> crate::Result<BTreeMap<AccountKeyPurposeId, KeychainUsageState>> {
                self.storage
                    .get::<db::DBKeychainUsageStates, _>()
                    .prefix_iter_decoded(account_id)
                    .map_err(crate::Error::from)
                    .map(Iterator::collect)
            }

            fn get_public_key(
                &self,
                id: &AccountDerivationPathId,
            ) -> crate::Result<Option<ExtendedPublicKey>> {
                self.read::<db::DBPubKeys, _, _>(id)
            }

            fn get_public_keys(
                &self,
                account_id: &AccountId,
            ) -> crate::Result<BTreeMap<AccountDerivationPathId, ExtendedPublicKey>> {
                self.storage
                    .get::<db::DBPubKeys, _>()
                    .prefix_iter_decoded(account_id)
                    .map_err(crate::Error::from)
                    .map(Iterator::collect)
            }
        }

        impl<'st, B: storage::Backend> $TxType<'st, B> {
            // Read a value from the database and decode it
            fn read<DbMap, I, K>(&self, key: K) -> crate::Result<Option<DbMap::Value>>
            where
                DbMap: schema::DbMap,
                Schema: schema::HasDbMap<DbMap, I>,
                K: EncodeLike<DbMap::Key>,
            {
                let map = self.storage.get::<DbMap, I>();
                map.get(key).map_err(crate::Error::from).map(|x| x.map(|x| x.decode()))
            }

            // Read a value for a well-known entry
            fn read_value<E: well_known::Entry>(&self) -> crate::Result<Option<E::Value>> {
                self.read::<db::DBValue, _, _>(E::KEY).map(|x| {
                    x.map(|x| {
                        E::Value::decode_all(&mut x.as_ref())
                            .expect("db values to be encoded correctly")
                    })
                })
            }
        }
    };
}

impl_read_ops!(StoreTxRo);
impl_read_ops!(StoreTxRw);
impl_read_ops!(StoreTxRoUnlocked);
impl_read_ops!(StoreTxRwUnlocked);

impl<'st, B: storage::Backend> WalletStorageEncryptionRead for StoreTxRo<'st, B> {
    fn get_encryption_key_kdf_challenge(&self) -> crate::Result<Option<KdfChallenge>> {
        self.read_value::<well_known::EncryptionKeyKdfChallenge>()
    }

    /// Check if the provided encryption_key can decrypt all of the root keys
    fn check_can_decrypt_all_root_keys(&self, encryption_key: &SymmetricKey) -> crate::Result<()> {
        self.storage
            .get::<db::DBRootKeys, _>()
            .prefix_iter_decoded(&())
            .map_err(crate::Error::from)
            .map(|mut item| {
                item.try_for_each(|(_, v)| {
                    let _decrypted_value =
                        v.try_decrypt_then_take(encryption_key).map_err(|err| match err {
                            MaybeEncryptedError::DecryptionError(_) => {
                                crate::Error::WalletInvalidPassword
                            }
                            MaybeEncryptedError::DecodingError(err) => {
                                panic!("corrupted DB error in decoding of root keys: {}", err)
                            }
                        })?;

                    Ok(())
                })
            })?
    }
}
macro_rules! impl_read_unlocked_ops {
    ($TxType:ident) => {
        /// Wallet data storage transaction
        impl<'st, B: storage::Backend> WalletStorageReadUnlocked for $TxType<'st, B> {
            fn get_root_key(&self, id: &RootKeyId) -> crate::Result<Option<RootKeyContent>> {
                Ok(self.read::<db::DBRootKeys, _, _>(id)?.map(|v| {
                    v.try_take(self.encryption_key).expect("key was checked when unlocked")
                }))
            }

            /// Collect and return all keys from the storage
            fn get_all_root_keys(&self) -> crate::Result<BTreeMap<RootKeyId, RootKeyContent>> {
                self.storage
                    .get::<db::DBRootKeys, _>()
                    .prefix_iter_decoded(&())
                    .map_err(crate::Error::from)
                    .map(|item| {
                        item.map(|(k, v)| {
                            (
                                k,
                                v.try_take(self.encryption_key)
                                    .expect("key was checked when unlocked"),
                            )
                        })
                    })
                    .map(Iterator::collect)
            }
        }
    };
}

impl_read_unlocked_ops!(StoreTxRoUnlocked);
impl_read_unlocked_ops!(StoreTxRwUnlocked);

macro_rules! impl_write_ops {
    ($TxType:ident) => {
        /// Wallet data storage transaction
        impl<'st, B: storage::Backend> WalletStorageWriteLocked for $TxType<'st, B> {
            fn set_storage_version(&mut self, version: u32) -> crate::Result<()> {
                self.write_value::<well_known::StoreVersion>(&version)
            }

            fn set_transaction(
                &mut self,
                id: &AccountWalletTxId,
                tx: &WalletTx,
            ) -> crate::Result<()> {
                self.write::<db::DBTxs, _, _, _>(id, tx)
            }

            fn del_transaction(&mut self, id: &AccountWalletTxId) -> crate::Result<()> {
                self.storage.get_mut::<db::DBTxs, _>().del(id).map_err(Into::into)
            }

            // fn set_block(
            //     &mut self,
            //     block_height: &AccountBlockHeight,
            //     block: &WalletBlock,
            // ) -> crate::Result<()> {
            //     self.write::<db::DBBlocks, _, _, _>(block_height, block)
            // }

            // fn del_block(&mut self, block_height: &AccountBlockHeight) -> crate::Result<()> {
            //     self.storage.get_mut::<db::DBBlocks, _>().del(block_height).map_err(Into::into)
            // }

            // fn set_transaction(&mut self, id: &AccountTxId, tx: &WalletTx) -> crate::Result<()> {
            //     self.write::<db::DBTxs, _, _, _>(id, tx)
            // }

            // fn del_transaction(&mut self, id: &AccountTxId) -> crate::Result<()> {
            //     self.storage.get_mut::<db::DBTxs, _>().del(id).map_err(Into::into)
            // }

            fn set_account(&mut self, id: &AccountId, tx: &AccountInfo) -> crate::Result<()> {
                self.write::<db::DBAccounts, _, _, _>(id, tx)
            }

            fn del_account(&mut self, id: &AccountId) -> crate::Result<()> {
                self.storage.get_mut::<db::DBAccounts, _>().del(id).map_err(Into::into)
            }

            fn set_address(
                &mut self,
                id: &AccountDerivationPathId,
                address: &Address,
            ) -> crate::Result<()> {
                self.write::<db::DBAddresses, _, _, _>(id, address)
            }

            fn del_address(&mut self, id: &AccountDerivationPathId) -> crate::Result<()> {
                self.storage.get_mut::<db::DBAddresses, _>().del(id).map_err(Into::into)
            }

            fn set_keychain_usage_state(
                &mut self,
                id: &AccountKeyPurposeId,
                usage_state: &KeychainUsageState,
            ) -> crate::Result<()> {
                self.write::<db::DBKeychainUsageStates, _, _, _>(id, usage_state)
            }

            fn del_keychain_usage_state(&mut self, id: &AccountKeyPurposeId) -> crate::Result<()> {
                self.storage
                    .get_mut::<db::DBKeychainUsageStates, _>()
                    .del(id)
                    .map_err(Into::into)
            }

            fn set_public_key(
                &mut self,
                id: &AccountDerivationPathId,
                pub_key: &ExtendedPublicKey,
            ) -> crate::Result<()> {
                self.write::<db::DBPubKeys, _, _, _>(id, pub_key)
            }
            fn det_public_key(&mut self, id: &AccountDerivationPathId) -> crate::Result<()> {
                self.storage.get_mut::<db::DBPubKeys, _>().del(id).map_err(Into::into)
            }
        }

        impl<'st, B: storage::Backend> $TxType<'st, B> {
            // Encode a value and write it to the database
            fn write<DbMap, I, K, V>(&mut self, key: K, value: V) -> crate::Result<()>
            where
                DbMap: schema::DbMap,
                Schema: schema::HasDbMap<DbMap, I>,
                K: EncodeLike<<DbMap as schema::DbMap>::Key>,
                V: EncodeLike<<DbMap as schema::DbMap>::Value>,
            {
                self.storage.get_mut::<DbMap, I>().put(key, value).map_err(Into::into)
            }

            // Write a value for a well-known entry
            fn write_value<E: well_known::Entry>(&mut self, val: &E::Value) -> crate::Result<()> {
                self.write::<db::DBValue, _, _, _>(E::KEY, val.encode())
            }
        }
    };
}

impl_write_ops!(StoreTxRw);
impl_write_ops!(StoreTxRwUnlocked);

impl<'st, B: storage::Backend> WalletStorageEncryptionWrite for StoreTxRwUnlocked<'st, B> {
    fn set_encryption_kdf_challenge(&mut self, salt: &KdfChallenge) -> crate::Result<()> {
        self.write_value::<well_known::EncryptionKeyKdfChallenge>(salt)
            .map_err(Into::into)
    }

    fn encrypt_root_keys(
        &mut self,
        new_encryption_key: &Option<SymmetricKey>,
    ) -> crate::Result<()> {
        let changed_root_keys: Vec<_> = self
            .storage
            .get::<db::DBRootKeys, _>()
            .prefix_iter_decoded(&())?
            .map(|(k, v)| {
                let decrypted =
                    v.try_take(self.encryption_key).expect("key was checked when unlocked");
                (k, MaybeEncrypted::new(&decrypted, new_encryption_key))
            })
            .collect();

        changed_root_keys
            .into_iter()
            .try_for_each(|(k, v)| self.write::<db::DBRootKeys, _, _, _>(k, v))
    }
}

/// Wallet data storage transaction
impl<'st, B: storage::Backend> WalletStorageWriteUnlocked for StoreTxRwUnlocked<'st, B> {
    fn set_root_key(&mut self, id: &RootKeyId, tx: &RootKeyContent) -> crate::Result<()> {
        let value = MaybeEncrypted::new(tx, self.encryption_key);
        self.write::<db::DBRootKeys, _, _, _>(id, value)
    }

    fn del_root_key(&mut self, id: &RootKeyId) -> crate::Result<()> {
        self.storage.get_mut::<db::DBRootKeys, _>().del(id).map_err(Into::into)
    }
}

impl<'st, B: storage::Backend> crate::TransactionRoLocked for StoreTxRo<'st, B> {
    fn close(self) {
        self.storage.close()
    }
}

impl<'st, B: storage::Backend> crate::TransactionRoUnlocked for StoreTxRoUnlocked<'st, B> {
    fn close(self) {
        self.storage.close()
    }
}

impl<'st, B: storage::Backend> crate::TransactionRwLocked for StoreTxRw<'st, B> {
    fn commit(self) -> crate::Result<()> {
        self.storage.commit().map_err(Into::into)
    }

    fn abort(self) {
        self.storage.abort()
    }
}

impl<'st, B: storage::Backend> crate::TransactionRwUnlocked for StoreTxRwUnlocked<'st, B> {
    fn commit(self) -> crate::Result<()> {
        self.storage.commit().map_err(Into::into)
    }

    fn abort(self) {
        self.storage.abort()
    }
}

impl<'st, B: storage::Backend> crate::IsTransaction for StoreTxRo<'st, B> {}
impl<'st, B: storage::Backend> crate::IsTransaction for StoreTxRw<'st, B> {}
impl<'st, B: storage::Backend> crate::IsTransaction for StoreTxRoUnlocked<'st, B> {}
impl<'st, B: storage::Backend> crate::IsTransaction for StoreTxRwUnlocked<'st, B> {}
