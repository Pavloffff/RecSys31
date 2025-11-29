import json

from .abstract_codec import AbstractCodec


class JsonCodec(AbstractCodec):
    def decode(self, message: bytes, encoding: str) -> dict:
        return json.loads(message.decode(encoding))

    def encode(self, message: dict, encoding: str) -> bytes:
        return json.dumps(message).encode(encoding)

