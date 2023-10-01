from pydantic import BaseModel


class RequestParams(BaseModel):
    message: str
