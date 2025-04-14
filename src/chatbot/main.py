# src/chatbot/main.py
# Este arquivo é o ponto de entrada da aplicação FastAPI.

from fastapi import FastAPI
from src.chatbot.api.routes import router

app = FastAPI()

# Incluindo as rotas da API
app.include_router(router)
