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

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use async_trait::async_trait;
use futures::{future::BoxFuture, stream::FuturesUnordered, StreamExt};
use tokio::net::{TcpListener, TcpStream};

use crate::{
    net::{
        default_backend::transport::{
            traits::TransportAddress, PeerStream, TransportListener, TransportSocket,
        },
        AsBannableAddress,
    },
    peer_manager::global_ip::IsGlobalIp,
    types::peer_address::PeerAddress,
    Result,
};

impl TransportAddress for SocketAddr {
    fn as_peer_address(&self) -> PeerAddress {
        (*self).into()
    }

    fn from_peer_address(address: &PeerAddress, allow_private_ips: bool) -> Option<Self> {
        match &address {
            PeerAddress::Ip4(socket)
                if (Ipv4Addr::from(socket.ip).is_global_unicast_ip() || allow_private_ips)
                    && socket.port != 0 =>
            {
                Some(address.into())
            }
            PeerAddress::Ip6(socket)
                if (Ipv6Addr::from(socket.ip).is_global_unicast_ip() || allow_private_ips)
                    && socket.port != 0 =>
            {
                Some(address.into())
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct TcpTransportSocket;

impl TcpTransportSocket {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl TransportSocket for TcpTransportSocket {
    type Address = SocketAddr;
    type BannableAddress = IpAddr;
    type Listener = TcpTransportListener;
    type Stream = TcpTransportStream;

    async fn bind(&self, addresses: Vec<Self::Address>) -> Result<Self::Listener> {
        TcpTransportListener::new(addresses)
    }

    fn connect(&self, address: Self::Address) -> BoxFuture<'static, Result<Self::Stream>> {
        Box::pin(async move {
            let stream = TcpStream::connect(address).await?;
            Ok(stream)
        })
    }
}

pub struct TcpTransportListener {
    listeners: Vec<TcpListener>,
}

impl TcpTransportListener {
    fn new(addresses: Vec<SocketAddr>) -> Result<Self> {
        let listeners = addresses
            .into_iter()
            .map(|address| -> Result<TcpListener> {
                // Use socket2 crate because we need consistent behavior between platforms.
                // See https://github.com/tokio-rs/tokio-core/issues/227
                let socket = socket2::Socket::new(
                    socket2::Domain::for_address(address),
                    socket2::Type::STREAM,
                    None,
                )?;

                socket.set_nonblocking(true)?;

                if address.is_ipv6() {
                    // When IPV6_V6ONLY is disabled listening IPv6 socket will also accept incoming connections from IPv4.
                    // Remote address will be reported as IPv4 mapped to IPv6 (for example ::ffff:192.168.1.2).
                    // Enable IPV6_V6ONLY explicitly because default value differs between platforms
                    // (true on windows and false on most other OSs).
                    // Bitcoin and libp2p work same way.
                    socket.set_only_v6(true)?;
                }

                // Allow faster app restarts on *nix (same way it's done in tokio/mio)
                #[cfg(not(windows))]
                socket.set_reuse_address(true)?;

                socket.bind(&address.into())?;

                // Set max count of pending TCP connections, we don't need a lot
                socket.listen(32)?;

                let listener = TcpListener::from_std(socket.into())?;

                Ok(listener)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { listeners })
    }
}

#[async_trait]
impl TransportListener for TcpTransportListener {
    type Stream = TcpTransportStream;
    type Address = SocketAddr;

    async fn accept(&mut self) -> Result<(TcpTransportStream, SocketAddr)> {
        // select_next_some will panic if polled while empty
        if self.listeners.is_empty() {
            return std::future::pending().await;
        }
        let mut tasks: FuturesUnordered<_> =
            self.listeners.iter().map(|listener| listener.accept()).collect();
        let (stream, address) = tasks.select_next_some().await?;
        Ok((stream, address))
    }

    fn local_addresses(&self) -> Result<Vec<SocketAddr>> {
        let local_addr = self
            .listeners
            .iter()
            .map(|listener| listener.local_addr())
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(local_addr)
    }
}

impl AsBannableAddress for SocketAddr {
    type BannableAddress = IpAddr;

    fn as_bannable(&self) -> Self::BannableAddress {
        self.ip()
    }
}

pub type TcpTransportStream = TcpStream;

impl PeerStream for TcpTransportStream {}

#[cfg(test)]
mod tests {
    use common::{
        chain::block::Block,
        primitives::{Id, H256},
    };
    use crypto::random::Rng;
    use test_utils::random::Seed;

    use crate::{
        message::BlockListRequest,
        testing_utils::{TestTransportMaker, TestTransportTcp},
    };

    use super::*;
    use crate::net::default_backend::{transport::BufferedTranscoder, types::Message};

    #[rstest::rstest]
    #[trace]
    #[case(Seed::from_entropy())]
    #[tokio::test]
    async fn send_recv(#[case] seed: Seed) {
        let mut rng = test_utils::random::make_seedable_rng(seed);

        let transport = TcpTransportSocket::new();
        let mut server = transport.bind(vec![TestTransportTcp::make_address()]).await.unwrap();
        let peer_fut = transport.connect(server.local_addresses().unwrap()[0]);

        let (server_res, peer_res) = tokio::join!(server.accept(), peer_fut);
        let server_stream = server_res.unwrap().0;
        let peer_stream = peer_res.unwrap();

        let message = Message::BlockListRequest(BlockListRequest::new(vec![]));
        let mut peer_stream = BufferedTranscoder::new(peer_stream, rng.gen_range(128..1024));
        peer_stream.send(message.clone()).await.unwrap();

        let mut server_stream = BufferedTranscoder::new(server_stream, rng.gen_range(128..1024));
        assert_eq!(server_stream.recv().await.unwrap(), message);
    }

    #[rstest::rstest]
    #[trace]
    #[case(Seed::from_entropy())]
    #[tokio::test]
    async fn send_2_reqs(#[case] seed: Seed) {
        let mut rng = test_utils::random::make_seedable_rng(seed);

        let transport = TcpTransportSocket::new();
        let mut server = transport.bind(vec![TestTransportTcp::make_address()]).await.unwrap();
        let peer_fut = transport.connect(server.local_addresses().unwrap()[0]);

        let (server_res, peer_res) = tokio::join!(server.accept(), peer_fut);
        let server_stream = server_res.unwrap().0;
        let peer_stream = peer_res.unwrap();

        let message_1 = Message::BlockListRequest(BlockListRequest::new(vec![]));
        let id: Id<Block> = H256::random_using(&mut rng).into();
        let message_2 = Message::BlockListRequest(BlockListRequest::new(vec![id]));

        let mut peer_stream = BufferedTranscoder::new(peer_stream, rng.gen_range(512..2048));
        peer_stream.send(message_1.clone()).await.unwrap();
        peer_stream.send(message_2.clone()).await.unwrap();

        let mut server_stream = BufferedTranscoder::new(server_stream, rng.gen_range(512..2048));
        assert_eq!(server_stream.recv().await.unwrap(), message_1);
        assert_eq!(server_stream.recv().await.unwrap(), message_2);
    }
}
