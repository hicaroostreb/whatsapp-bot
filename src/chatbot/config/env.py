# src/chatbot/config/env.py
# Este arquivo contém a configuração de variáveis de ambiente para o chatbot.

import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()


class Config:
    # URLs e chaves de API
    WAHA_API_URL = os.getenv("WAHA_API_URL", "http://localhost:3000")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
