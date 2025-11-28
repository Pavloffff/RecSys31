import json

from services.codec.abstract_codec import AbstractCodec


class JsonCodec(AbstractCodec):
    def decode(self, message: bytes, encoding: str) -> dict:
        return json.loads(message.decode(encoding))
        