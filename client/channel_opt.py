import struct


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


keepalive_options = KeepAliveOptions()
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
