from pydantic import BaseModel


class Request(BaseModel):
    token: str
    request: str
    