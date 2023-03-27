import asyncio
import logging
from loguru import logger
import numpy as np
import rapidjson as json
from google.protobuf.json_format import MessageToJson


import grpc
import dec_pb2
import dec_pb2_grpc

from client.channel_opt import channel_opt


async def run(shape, data, url, channel_opt):

    async with grpc.aio.insecure_channel(url, options=channel_opt) as channel:
        stub = dec_pb2_grpc.DecoderStub(channel)
        response = await stub.Decode(dec_pb2.Logits(shape=shape, data=data))
        response = json.loads(
            MessageToJson(response, preserving_proto_field_name=True))
        return response


if __name__ == '__main__':
    url = "localhost:1508"

    sample_data = np.random.rand(108, 32)
    sample_shape = sample_data.shape

    logging.basicConfig()

    sample_response = asyncio.run(run(sample_shape, sample_data.flatten(),url, channel_opt))

    logger.debug(sample_response)

    
