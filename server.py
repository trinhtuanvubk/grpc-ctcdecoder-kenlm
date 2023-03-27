import asyncio
import logging
from loguru import logger
import base64
import numpy as np
from google.protobuf.json_format import MessageToJson


import grpc
import dec_pb2
import dec_pb2_grpc

from server.helpers import decode_logits

class Decoder(dec_pb2_grpc.DecoderServicer):

    async def Decode(
        self,
        request: dec_pb2.Logits,
        context: grpc.aio.ServicerContext,
    ) -> dec_pb2.Transcription:

        logger.debug(request)
        shape = request.shape 
        data = np.array(request.data)
        data = np.reshape(data, shape)
        gt, bt, bdo = decode_logits(data)
        logger.debug(gt)
        logger.debug(bt)
        logger.debug(bdo)
        return dec_pb2.Transcription(greedy_trans=gt,
                                     beam_trans=bt,
                                     beam_decoded_offsets=bdo)


async def serve(listen_addr) -> None:
    server = grpc.aio.server()
    dec_pb2_grpc.add_DecoderServicer_to_server(Decoder(), server)
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


if __name__ == '__main__':
    listen_addr = '[::]:1508'

    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(listen_addr))
