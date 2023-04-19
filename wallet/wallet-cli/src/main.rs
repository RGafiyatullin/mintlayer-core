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

use std::{path::PathBuf, sync::Arc};

use clap::Parser;
use common::chain::ChainConfig;
use dialoguer::theme::ColorfulTheme;
use node_comm::{make_rpc_client, node_traits::NodeInterface};
use serialization::hex::HexEncode;
use utils::default_data_dir::PrepareDataDirError;
use wallet::{Wallet, WalletError};

mod config;
mod repl;

// TODO(PR): Add context
#[macro_export]
macro_rules! cli_println {
    ($($arg:tt)*) => { ::std::println!($($arg)*) };
}

#[derive(thiserror::Error, Debug)]
pub enum WalletCliError {
    #[error("RPC error: {0}")]
    RpcError(node_comm::rpc_client::NodeRpcError),
    #[error("Wallet error: {0}")]
    WalletError(wallet::wallet::WalletError),
    #[error("Console IO error: {0}")]
    ConsoleIoError(std::io::Error),
    #[error("Cookie file {0} read error: {1}. Please make sure the node is started.")]
    CookieFileReadError(PathBuf, std::io::Error),
    #[error("Prepare data dir error: {0}")]
    PrepareData(PrepareDataDirError),
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
    #[error("Cancelled")]
    Cancelled,
}

#[derive(Clone, Copy)]
enum ImportMnemonic {
    Generate,
    Import,
    Cancel,
}

impl Into<&str> for ImportMnemonic {
    fn into(self) -> &'static str {
        match self {
            ImportMnemonic::Generate => "Generate new mnemonic",
            ImportMnemonic::Import => "Import mnemonic",
            ImportMnemonic::Cancel => "Cancel",
        }
    }
}

fn select_helper<T: Clone + Into<&'static str>>(
    theme: &ColorfulTheme,
    prompt: &str,
    items: &[T],
) -> Result<T, WalletCliError> {
    let texts = items.iter().cloned().map(Into::into).collect::<Vec<&str>>();
    let index = dialoguer::Select::with_theme(theme)
        .with_prompt(prompt)
        .default(0)
        .items(&texts)
        .interact_opt()
        .map_err(WalletCliError::ConsoleIoError)?
        .ok_or(WalletCliError::Cancelled)?;
    Ok(items[index].clone())
}

fn new_wallet(
    chain_config: Arc<ChainConfig>,
    db: Arc<wallet_storage::Store<wallet_storage::DefaultBackend>>,
    theme: &ColorfulTheme,
) -> Result<Wallet<wallet_storage::DefaultBackend>, WalletCliError> {
    let action = select_helper(
        theme,
        "Wallet is not initialized",
        &[ImportMnemonic::Generate, ImportMnemonic::Import, ImportMnemonic::Cancel],
    )?;

    let mnemonic: String = match action {
        ImportMnemonic::Generate => {
            let new_mnemonoc = wallet::wallet::generate_new_mnemonic();
            cli_println!("New mnemonic: {}", new_mnemonoc.to_string());
            cli_println!("Please write it somewhere safe to be able to restore your wallet.");
            new_mnemonoc.to_string()
        }
        ImportMnemonic::Import => dialoguer::Input::with_theme(theme)
            .with_prompt("Mnemonic")
            .interact_text()
            .map_err(WalletCliError::ConsoleIoError)?,
        ImportMnemonic::Cancel => return Err(WalletCliError::Cancelled),
    };

    // TODO: Add optional passphrase

    Wallet::new_wallet(Arc::clone(&chain_config), db, &mnemonic, None)
        .map_err(WalletCliError::WalletError)
}

async fn run() -> Result<(), WalletCliError> {
    logging::init_logging::<&std::path::Path>(None);

    let args = config::WalletCliArgs::parse();
    let config = config::WalletCliConfig::from_args(args)?;
    let chain_config = Arc::new(common::chain::config::Builder::new(config.chain_type).build());

    let db = wallet::wallet::open_or_create_wallet_file(&config.wallet_file)
        .map_err(WalletCliError::WalletError)?;

    let theme = ColorfulTheme::default();

    let _wallet = match Wallet::load_wallet(Arc::clone(&chain_config), Arc::clone(&db)) {
        Ok(wallet) => wallet,
        Err(WalletError::WalletNotInitialized) => new_wallet(chain_config, db, &theme)?,
        Err(e) => return Err(WalletCliError::WalletError(e)),
    };

    let rpc_client = make_rpc_client(
        config.rpc_address,
        Some((&config.rpc_username, &config.rpc_password)),
    )
    .await
    .map_err(WalletCliError::RpcError)?;

    cli_println!(
        "Best block id: {}",
        rpc_client
            .get_best_block_id()
            .await
            .map(|block| block.hex_encode())
            .unwrap_or_else(|e| e.to_string())
    );

    repl::start_cli_repl()
}

#[tokio::main]
async fn main() {
    run().await.unwrap_or_else(|err| {
        cli_println!("wallet-cli launch failed: {err}");
        std::process::exit(1)
    })
}
