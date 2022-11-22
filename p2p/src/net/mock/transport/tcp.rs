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

use std::{
    io,
    net::{IpAddr, SocketAddr},
};

mod encryption;

use async_trait::async_trait;
use bytes::{Buf, BytesMut};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};
use tokio_util::codec::{Decoder, Encoder};

use serialization::{Decode, Encode};

use crate::{
    constants::MAX_MESSAGE_SIZE,
    net::{
        mock::{
            transport::{MockListener, MockStream, MockTransport},
            types::Message,
        },
        AsBannableAddress, IsBannableAddress,
    },
    P2pError, Result,
};

use self::encryption::{Encryption, NoiseEncryption};

#[derive(Debug)]
pub struct TcpMockTransportBase<E: Encryption>(std::marker::PhantomData<E>);

// By default the transport uses Noise protocol encryption
pub type TcpMockTransport = TcpMockTransportBase<NoiseEncryption>;

#[async_trait]
impl<E: Encryption + 'static> MockTransport for TcpMockTransportBase<E> {
    type Address = SocketAddr;
    type BannableAddress = IpAddr;
    type Listener = TcpMockListener<E>;
    type Stream = TcpMockStream<E>;

    async fn bind(address: Self::Address) -> Result<Self::Listener> {
        let listener = TcpListener::bind(address).await?;
        Ok(TcpMockListener(listener, Default::default()))
    }

    async fn connect(address: Self::Address) -> Result<Self::Stream> {
        let tcp_stream = TcpStream::connect(address).await?;
        let noise_stream = TcpMockStream::new(tcp_stream, Side::Outbound).await?;
        Ok(noise_stream)
    }
}

pub struct TcpMockListener<E: Encryption>(TcpListener, std::marker::PhantomData<E>);

#[async_trait]
impl<E: Encryption> MockListener<TcpMockStream<E>, SocketAddr> for TcpMockListener<E> {
    async fn accept(&mut self) -> Result<(TcpMockStream<E>, SocketAddr)> {
        let (tcp_stream, address) = TcpListener::accept(&self.0).await?;
        let noise_stream = TcpMockStream::new(tcp_stream, Side::Inbound).await?;
        Ok((noise_stream, address))
    }

    fn local_address(&self) -> Result<SocketAddr> {
        self.0.local_addr().map_err(Into::into)
    }
}

pub struct TcpMockStream<E: Encryption> {
    stream: E::Stream,
    buffer: BytesMut,
}

pub enum Side {
    Inbound,
    Outbound,
}

impl<E: Encryption> TcpMockStream<E> {
    async fn new(base: TcpStream, side: Side) -> Result<Self> {
        let stream = E::handshake(base, side).await?;
        Ok(Self {
            stream,
            buffer: BytesMut::new(),
        })
    }
}

#[async_trait]
impl<E: Encryption> MockStream for TcpMockStream<E> {
    async fn send(&mut self, msg: Message) -> Result<()> {
        let mut buf = bytes::BytesMut::new();
        EncoderDecoder {}.encode(msg, &mut buf)?;
        self.stream.write_all(&buf).await?;
        self.stream.flush().await?;
        Ok(())
    }

    /// Read a framed message from socket
    ///
    /// First try to decode whatever may be in the stream's buffer and if it's empty
    /// or the frame hasn't been completely received, wait on the socket until the buffer
    /// has all data. If the buffer has a full frame that can be decoded, return that without
    /// calling the socket first.
    async fn recv(&mut self) -> Result<Option<Message>> {
        match (EncoderDecoder {}.decode(&mut self.buffer)) {
            Ok(None) => {
                if self.stream.read_buf(&mut self.buffer).await? == 0 {
                    return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
                }
                self.recv().await
            }
            frame => frame,
        }
    }
}

struct EncoderDecoder {}

impl Decoder for EncoderDecoder {
    type Item = Message;
    type Error = P2pError;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>> {
        if src.len() < 4 {
            return Ok(None);
        }

        let mut length_bytes = [0u8; 4];
        length_bytes.copy_from_slice(&src[..4]);
        let length = u32::from_le_bytes(length_bytes) as usize;

        if length > MAX_MESSAGE_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Frame of length {length} is too large"),
            )
            .into());
        }

        if src.len() < 4 + length {
            src.reserve(4 + length - src.len());
            return Ok(None);
        }

        let data = src[4..4 + length].to_vec();
        src.advance(4 + length);

        match Message::decode(&mut &data[..]) {
            Ok(msg) => Ok(Some(msg)),
            Err(e) => {
                Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()).into())
            }
        }
    }
}

impl Encoder<Message> for EncoderDecoder {
    type Error = P2pError;

    fn encode(&mut self, msg: Message, dst: &mut BytesMut) -> Result<()> {
        let encoded = msg.encode();

        if encoded.len() > MAX_MESSAGE_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Frame of length {} is too large", encoded.len()),
            )
            .into());
        }

        let len_slice = u32::to_le_bytes(encoded.len() as u32);

        dst.reserve(4 + encoded.len());
        dst.extend_from_slice(&len_slice);
        dst.extend_from_slice(&encoded);

        Ok(())
    }
}

impl AsBannableAddress for SocketAddr {
    type BannableAddress = IpAddr;

    fn as_bannable(&self) -> Self::BannableAddress {
        self.ip()
    }
}

impl IsBannableAddress for SocketAddr {
    fn is_bannable(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::{
        message::{BlockListRequest, Request},
        mock::types::MockRequestId,
    };

    #[tokio::test]
    async fn send_recv() {
        let address = "[::1]:0".parse().unwrap();
        let mut server = TcpMockTransport::bind(address).await.unwrap();
        let peer_fut = TcpMockTransport::connect(server.local_address().unwrap());

        let (server_res, peer_res) = tokio::join!(server.accept(), peer_fut);
        let mut server_stream = server_res.unwrap().0;
        let mut peer_stream = peer_res.unwrap();

        let request_id = MockRequestId::new(1337u64);
        let request = Request::BlockListRequest(BlockListRequest::new(vec![]));
        peer_stream
            .send(Message::Request {
                request_id,
                request: request.clone(),
            })
            .await
            .unwrap();

        assert_eq!(
            server_stream.recv().await.unwrap().unwrap(),
            Message::Request {
                request_id,
                request,
            }
        );
    }

    #[tokio::test]
    async fn send_2_reqs() {
        let address = "[::1]:0".parse().unwrap();
        let mut server = TcpMockTransport::bind(address).await.unwrap();
        let peer_fut = TcpMockTransport::connect(server.local_address().unwrap());

        let (server_res, peer_res) = tokio::join!(server.accept(), peer_fut);
        let mut server_stream = server_res.unwrap().0;
        let mut peer_stream = peer_res.unwrap();

        let id_1 = MockRequestId::new(1337u64);
        let request = Request::BlockListRequest(BlockListRequest::new(vec![]));
        peer_stream
            .send(Message::Request {
                request_id: id_1,
                request: request.clone(),
            })
            .await
            .unwrap();

        let id_2 = MockRequestId::new(1338u64);
        peer_stream
            .send(Message::Request {
                request_id: id_2,
                request: request.clone(),
            })
            .await
            .unwrap();

        assert_eq!(
            server_stream.recv().await.unwrap().unwrap(),
            Message::Request {
                request_id: id_1,
                request: request.clone(),
            }
        );
        assert_eq!(
            server_stream.recv().await.unwrap().unwrap(),
            Message::Request {
                request_id: id_2,
                request,
            }
        );
    }
}
