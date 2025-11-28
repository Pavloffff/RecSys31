from pydantic import BaseModel


class Request(BaseModel):
    context: str
    text: str
    