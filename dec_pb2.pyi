from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Logits(_message.Message):
    __slots__ = ["data", "shape"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[float]
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, shape: _Optional[_Iterable[int]] = ..., data: _Optional[_Iterable[float]] = ...) -> None: ...

class Transcription(_message.Message):
    __slots__ = ["beam_decoded_offsets", "beam_trans", "greedy_trans"]
    BEAM_DECODED_OFFSETS_FIELD_NUMBER: _ClassVar[int]
    BEAM_TRANS_FIELD_NUMBER: _ClassVar[int]
    GREEDY_TRANS_FIELD_NUMBER: _ClassVar[int]
    beam_decoded_offsets: _containers.RepeatedScalarFieldContainer[int]
    beam_trans: str
    greedy_trans: str
    def __init__(self, greedy_trans: _Optional[str] = ..., beam_trans: _Optional[str] = ..., beam_decoded_offsets: _Optional[_Iterable[int]] = ...) -> None: ...
