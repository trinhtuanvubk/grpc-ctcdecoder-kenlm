import asyncio

import logging

import grpc
import dec_pb2
import dec_pb2_grpc

import struct
import rapidjson as json
from google.protobuf.json_format import MessageToJson


INT32_MAX = 2**(struct.Struct('i').size * 8 - 1) - 1
MAX_GRPC_MESSAGE_SIZE = INT32_MAX




class KeepAliveOptions:
    """A KeepAliveOptions object is used to encapsulate GRPC KeepAlive
    related parameters for initiating an InferenceServerclient object.

    See the https://github.com/grpc/grpc/blob/master/doc/keepalive.md
    documentation for more information.

    Parameters
    ----------
    keepalive_time_ms: int
        The period (in milliseconds) after which a keepalive ping is sent on
        the transport. Default is INT32_MAX.

    keepalive_timeout_ms: int
        The period (in milliseconds) the sender of the keepalive ping waits
        for an acknowled
        gement. If it does not receive an acknowledgment
        within this time, it will close the connection. Default is 20000
        (20 seconds).

    keepalive_permit_without_calls: bool
        Allows keepalive pings to be sent even if there are no calls in flight.
        Default is False.

    http2_max_pings_without_data: int
        The maximum number of pings that can be sent when there is no
        data/header frame to be sent. gRPC Core will not continue sending
        pings if we run over the limit. Setting it to 0 allows sending pings
        without such a restriction. Default is 2.

    """

    def __init__(self,
                 keepalive_time_ms=INT32_MAX,
                 keepalive_timeout_ms=20000,
                 keepalive_permit_without_calls=False,
                 http2_max_pings_without_data=2):
        self.keepalive_time_ms = keepalive_time_ms
        self.keepalive_timeout_ms = keepalive_timeout_ms
        self.keepalive_permit_without_calls = keepalive_permit_without_calls
        self.http2_max_pings_without_data = http2_max_pings_without_data


class InferInput:
    """An object of InferInput class is used to describe
    input tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of input whose data will be described by this object
    shape : list
        The shape of the associated input.
    datatype : str
        The datatype of the associated input.

    """

    def __init__(self, name, shape, datatype):
        self._input = service_pb2.ModelInferRequest().InferInputTensor()
        self._input.name = name
        self._input.ClearField('shape')
        self._input.shape.extend(shape)
        self._input.datatype = datatype
        self._raw_content = None

    def name(self):
        """Get the name of input associated with this object.

        Returns
        -------
        str
            The name of input
        """
        return self._input.name

    def datatype(self):
        """Get the datatype of input associated with this object.

        Returns
        -------
        str
            The datatype of input
        """
        return self._input.datatype

    def shape(self):
        """Get the shape of input associated with this object.

        Returns
        -------
        list
            The shape of input
        """
        return self._input.shape

    def set_shape(self, shape):
        """Set the shape of input.

        Parameters
        ----------
        shape : list
            The shape of the associated input.
        """
        self._input.ClearField('shape')
        self._input.shape.extend(shape)

    def set_data_from_numpy(self, input_tensor):
        """Set the tensor data from the specified numpy array for
        input associated with this object.

        Parameters
        ----------
        input_tensor : numpy array
            The tensor data in numpy array format

        Raises
        ------
        InferenceServerException
            If failed to set data for the tensor.
        """
        if not isinstance(input_tensor, (np.ndarray,)):
            raise_error("input_tensor must be a numpy array")
        dtype = np_to_triton_dtype(input_tensor.dtype)
        if self._input.datatype != dtype:
            raise_error(
                "got unexpected datatype {} from numpy array, expected {}".
                format(dtype, self._input.datatype))
        valid_shape = True
        if len(self._input.shape) != len(input_tensor.shape):
            valid_shape = False
        for i in range(len(self._input.shape)):
            if self._input.shape[i] != input_tensor.shape[i]:
                valid_shape = False
        if not valid_shape:
            raise_error(
                "got unexpected numpy array shape [{}], expected [{}]".format(
                    str(input_tensor.shape)[1:-1],
                    str(self._input.shape)[1:-1]))

        self._input.parameters.pop('shared_memory_region', None)
        self._input.parameters.pop('shared_memory_byte_size', None)
        self._input.parameters.pop('shared_memory_offset', None)

        if self._input.datatype == "BYTES":
            serialized_output = serialize_byte_tensor(input_tensor)
            if serialized_output.size > 0:
                self._raw_content = serialized_output.item()
            else:
                self._raw_content = b''
        else:
            self._raw_content = input_tensor.tobytes()

    def set_shared_memory(self, region_name, byte_size, offset=0):
        """Set the tensor data from the specified shared memory region.

        Parameters
        ----------
        region_name : str
            The name of the shared memory region holding tensor data.
        byte_size : int
            The size of the shared memory region holding tensor data.
        offset : int
            The offset, in bytes, into the region where the data for
            the tensor starts. The default value is 0.

        """
        self._input.ClearField("contents")
        self._raw_content = None

        self._input.parameters[
            'shared_memory_region'].string_param = region_name
        self._input.parameters[
            'shared_memory_byte_size'].int64_param = byte_size
        if offset != 0:
            self._input.parameters['shared_memory_offset'].int64_param = offset

    def _get_tensor(self):
        """Retrieve the underlying InferInputTensor message.
        Returns
        -------
        protobuf message
            The underlying InferInputTensor protobuf message.
        """
        return self._input

    def _get_content(self):
        """Retrieve the contents for this tensor in raw bytes.
        Returns
        -------
        bytes
            The associated contents for this tensor in raw bytes.
        """
        return self._raw_content


class InferResult:
    """An object of InferResult class holds the response of
    an inference request and provide methods to retrieve
    inference results.

    Parameters
    ----------
    result : protobuf message
        The ModelInferResponse returned by the server
    """

    def __init__(self, result):
        self._result = result

    def as_numpy(self, name):
        """Get the tensor data for output associated with this object
        in numpy format

        Parameters
        ----------
        name : str
            The name of the output tensor whose result is to be retrieved.

        Returns
        -------
        numpy array
            The numpy array containing the response data for the tensor or
            None if the data for specified tensor name is not found.
        """
        index = 0
        for output in self._result.outputs:
            if output.name == name:
                shape = []
                for value in output.shape:
                    shape.append(value)

                datatype = output.datatype
                if index < len(self._result.raw_output_contents):
                    if datatype == 'BYTES':
                        # String results contain a 4-byte string length
                        # followed by the actual string characters. Hence,
                        # need to decode the raw bytes to convert into
                        # array elements.
                        np_array = deserialize_bytes_tensor(
                            self._result.raw_output_contents[index])
                    else:
                        np_array = np.frombuffer(
                            self._result.raw_output_contents[index],
                            dtype=triton_to_np_dtype(datatype))
                elif len(output.contents.bytes_contents) != 0:
                    np_array = np.array(output.contents.bytes_contents,
                                        copy=False)
                else:
                    np_array = np.empty(0)
                np_array = np_array.reshape(shape)
                return np_array
            else:
                index += 1
        return None

    def get_output(self, name, as_json=False):
        """Retrieves the InferOutputTensor corresponding to the
        named ouput.

        Parameters
        ----------
        name : str
            The name of the tensor for which Output is to be
            retrieved.
        as_json : bool
            If True then returns response as a json dict, otherwise
            as a protobuf message. Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Returns
        -------
        protobuf message or dict
            If a InferOutputTensor with specified name is present in
            ModelInferResponse then returns it as a protobuf messsage
            or dict, otherwise returns None.
        """
        for output in self._result.outputs:
            if output.name == name:
                if as_json:
                    return json.loads(
                        MessageToJson(output, preserving_proto_field_name=True))
                else:
                    return output

        return None

    def get_response(self, as_json=False):
        """Retrieves the complete ModelInferResponse as a
        json dict object or protobuf message

        Parameters
        ----------
        as_json : bool
            If True then returns response as a json dict, otherwise
            as a protobuf message. Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Returns
        -------
        protobuf message or dict
            The underlying ModelInferResponse as a protobuf message or dict.
        """
        if as_json:
            return json.loads(
                MessageToJson(self._result, preserving_proto_field_name=True))
        else:
            return self._result


class InferenceServerClient:
    """An InferenceServerClient object is used to perform any kind of
    communication with the InferenceServer using gRPC protocol. Most
    of the methods are thread-safe except start_stream, stop_stream
    and async_stream_infer. Accessing a client stream with different
    threads will cause undefined behavior.

    Parameters
    ----------
    url : str
        The inference server URL, e.g. 'localhost:8001'.

    verbose : bool
        If True generate verbose output. Default value is False.

    ssl : bool
        If True use SSL encrypted secure channel. Default is False.

    root_certificates : str
        File holding the PEM-encoded root certificates as a byte
        string, or None to retrieve them from a default location
        chosen by gRPC runtime. The option is ignored if `ssl`
        is False. Default is None.

    private_key : str
        File holding the PEM-encoded private key as a byte string,
        or None if no private key should be used. The option is
        ignored if `ssl` is False. Default is None.

    certificate_chain : str
        File holding PEM-encoded certificate chain as a byte string
        to use or None if no certificate chain should be used. The
        option is ignored if `ssl` is False. Default is None.

    creds: grpc.ChannelCredentials
        A grpc.ChannelCredentials object to use for the connection.
        The ssl, root_certificates, private_key and certificate_chain
        options will be ignored when using this option. Default is None.

    keepalive_options: KeepAliveOptions
        Object encapsulating various GRPC KeepAlive options. See
        the class definition for more information. Default is None.

    channel_args: List[Tuple]
        List of Tuple pairs ("key", value) to be passed directly to the GRPC
        channel as the channel_arguments. If this argument is provided, it is
        expected the channel arguments are correct and complete, and the
        keepalive_options parameter will be ignored since the corresponding
        keepalive channel arguments can be set directly in this parameter. See
        https://grpc.github.io/grpc/python/glossary.html#term-channel_arguments
        for more details. Default is None.

    Raises
    ------
    Exception
        If unable to create a client.

    """

    def __init__(self,
                 url,
                 verbose=False,
                 ssl=False,
                 root_certificates=None,
                 private_key=None,
                 certificate_chain=None,
                 creds=None,
                 keepalive_options=None,
                 channel_args=None):

        # Explicitly check "is not None" here to support passing an empty
        # list to specify setting no channel arguments.
        if channel_args is not None:
            channel_opt = channel_args
        else:
            # Use GRPC KeepAlive client defaults if unspecified
            if not keepalive_options:
                keepalive_options = KeepAliveOptions()

            # To specify custom channel_opt, see the channel_args parameter.
            channel_opt = [
                ('grpc.max_send_message_length', MAX_GRPC_MESSAGE_SIZE),
                ('grpc.max_receive_message_length', MAX_GRPC_MESSAGE_SIZE),
                ('grpc.keepalive_time_ms', keepalive_options.keepalive_time_ms),
                ('grpc.keepalive_timeout_ms',
                 keepalive_options.keepalive_timeout_ms),
                ('grpc.keepalive_permit_without_calls',
                 keepalive_options.keepalive_permit_without_calls),
                ('grpc.http2.max_pings_without_data',
                 keepalive_options.http2_max_pings_without_data),
            ]

        if creds:
            self._channel = grpc.secure_channel(
                url, creds, options=channel_opt)
        elif ssl:
            rc_bytes = pk_bytes = cc_bytes = None
            if root_certificates is not None:
                with open(root_certificates, 'rb') as rc_fs:
                    rc_bytes = rc_fs.read()
            if private_key is not None:
                with open(private_key, 'rb') as pk_fs:
                    pk_bytes = pk_fs.read()
            if certificate_chain is not None:
                with open(certificate_chain, 'rb') as cc_fs:
                    cc_bytes = cc_fs.read()
            creds = grpc.ssl_channel_credentials(root_certificates=rc_bytes,
                                                 private_key=pk_bytes,
                                                 certificate_chain=cc_bytes)
            self._channel = grpc.secure_channel(
                url, creds, options=channel_opt)
        else:
            self._channel = grpc.insecure_channel(url, options=channel_opt)
        self._client_stub = dec_pb2_grpc.DecoderStub(self._channel)
        self._verbose = verbose
        self._stream = None

    async def infer(self,
                    inputs,
                    request_id="",
                    sequence_id=0,
                    sequence_start=False,
                    sequence_end=False,
                    priority=0,
                    timeout=None,
                    client_timeout=None,
                    headers=None,
                    compression_algorithm=None):

        # async with grpc.aio.insecure_channel("host.docker.internal:1507") as channel:
        # stub = dec_pb2_grpc.DecoderStub(channel)
        async with self._channel as channel:
            response = await self._client_stub.Decode(dec_pb2.LogitsList(logits="test grpc"))
            print(response)
            result = InferResult(response)
            return result
        # print(response)
