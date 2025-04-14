# src/chatbot/services/waha.py
# Este arquivo contém a classe Waha, que é responsável por interagir com a API do WhatsApp através do Waha.

import requests
from src.chatbot.config.env import Config


class Waha:
    def __init__(self, api_url=None):
        # Se não passar um api_url, utiliza o da configuração
        self.api_url = api_url or Config.WAHA_API_URL

    def _send_request(self, method, endpoint, payload=None):
        """Função interna para enviar as requisições HTTP"""
        url = f"{self.api_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        try:
            if method.lower() == "post":
                response = requests.post(url, json=payload, headers=headers)
            elif method.lower() == "get":
                response = requests.get(url, headers=headers)
            else:
                raise ValueError("Método HTTP não suportado")

            # Verificar se a resposta foi bem-sucedida
            response.raise_for_status()  # Lança um erro se o status não for 2xx
            return response.json()  # Retorna a resposta JSON

        except requests.exceptions.RequestException as e:
            print(f"[ERRO] Falha ao fazer requisição: {e}")
            return {"error": str(e)}

    def send_message(self, chat_id, message):
        """Envia uma mensagem para o WhatsApp"""
        payload = {
            "session": "default",
            "chatId": chat_id,
            "text": message,
        }
        return self._send_request("post", "/api/sendText", payload)

    def get_history_messages(self, chat_id, limit):
        """Obtém o histórico de mensagens"""
        endpoint = (
            f"/api/default/chats/{chat_id}/messages?limit={limit}&downloadMedia=false"
        )
        return self._send_request("get", endpoint)

    def start_typing(self, chat_id):
        """Simula a digitação do bot"""
        payload = {
            "session": "default",
            "chatId": chat_id,
        }
        return self._send_request("post", "/api/startTyping", payload)

    def stop_typing(self, chat_id):
        """Interrompe a digitação do bot"""
        payload = {
            "session": "default",
            "chatId": chat_id,
        }
        return self._send_request("post", "/api/stopTyping", payload)
