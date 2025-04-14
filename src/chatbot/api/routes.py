# src/chatbot/api/routes.py
# Este arquivo contém as rotas da API FastAPI para o chatbot.

from fastapi import APIRouter, Depends
from src.chatbot.services.message_handler import MessageHandler
from src.chatbot.core.ai_bot import AIBot
from src.chatbot.services.waha import Waha
from src.chatbot.models.payload import WebhookRequest, WebhookResponse


# Instâncias dos objetos são criadas por dependências
def get_waha() -> Waha:
    return Waha()


def get_ai_bot() -> AIBot:
    return AIBot()


def get_message_handler(
    waha: Waha = Depends(get_waha), ai_bot: AIBot = Depends(get_ai_bot)
) -> MessageHandler:
    return MessageHandler(waha, ai_bot)


router = APIRouter()


# Rota para a raiz
@router.get("/")
async def root():
    return {"message": "API is running"}


@router.post("/chatbot/webhook/", response_model=WebhookResponse)
async def webhook(
    request: WebhookRequest,
    message_handler: MessageHandler = Depends(get_message_handler),
):
    """Endpoint para receber mensagens do WhatsApp"""

    # A validação do Pydantic já ocorreu automaticamente
    chat_id = request.payload.from_
    message_id = request.payload.id
    message_text = request.payload.body.strip()
    user_name = (
        request.payload._data.get("notifyName", "") if request.payload._data else ""
    )

    # Processa a mensagem através do MessageHandler
    response = message_handler.process_incoming_message(
        chat_id, message_id, message_text, user_name
    )

    return WebhookResponse(status="success", response=response)
