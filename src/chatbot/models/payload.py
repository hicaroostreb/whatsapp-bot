# src/chatbot/models/payload.py
# Este arquivo contém os modelos de dados utilizados na aplicação.

from pydantic import BaseModel
from typing import Optional


class PayloadData(BaseModel):
    from_: str
    id: str
    body: str
    _data: Optional[dict] = None


class WebhookRequest(BaseModel):
    payload: PayloadData


class WebhookResponse(BaseModel):
    status: str
    response: Optional[str] = None
