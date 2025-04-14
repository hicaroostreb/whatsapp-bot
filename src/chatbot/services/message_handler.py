# src/chatbot/services/message_handler.py
# Este módulo é responsável por gerenciar o processamento de mensagens recebidas

import time
from threading import Timer
from src.chatbot.services.waha import Waha
from src.chatbot.core.ai_bot import AIBot


class MessageHandler:
    def __init__(self, waha: Waha, ai_bot: AIBot):
        self.waha = waha
        self.ai_bot = ai_bot
        self.processed_messages = set()
        self.pending_messages = {}
        self.message_timers = {}
        self.message_histories = {}
        self.WAIT_TIME = 10  # Segundos para aguardar mais mensagens

    def process_incoming_message(self, chat_id, message_id, message_text, user_name):
        """Processa a mensagem recebida"""
        # Ignora mensagens duplicadas
        if message_id in self.processed_messages:
            return {"status": "ignored", "reason": "duplicate"}

        self.processed_messages.add(message_id)

        # Adiciona à fila de mensagens pendentes
        self.add_to_pending(chat_id, message_text, message_id, user_name)

        # Configura o timer para processar a mensagem depois de um tempo
        self.setup_processing_timer(chat_id)

        return {"status": "queued", "message": "Message queued for processing"}

    def add_to_pending(self, chat_id, message_text, message_id, user_name):
        """Adiciona a mensagem à lista de pendentes"""
        if chat_id not in self.pending_messages:
            self.pending_messages[chat_id] = []

        self.pending_messages[chat_id].append(
            {
                "text": message_text,
                "id": message_id,
                "user_name": user_name,
                "timestamp": time.time(),
            }
        )

    def setup_processing_timer(self, chat_id):
        """Configura ou reconfigura o timer para processamento"""
        if chat_id in self.message_timers and self.message_timers[chat_id]:
            self.message_timers[chat_id].cancel()

        # Cria um novo timer
        self.message_timers[chat_id] = Timer(
            self.WAIT_TIME, self.process_messages, args=[chat_id]
        )
        self.message_timers[chat_id].start()

    def process_messages(self, chat_id):
        """Processa as mensagens pendentes após o tempo de espera"""
        if chat_id not in self.pending_messages or not self.pending_messages[chat_id]:
            return

        messages = self.pending_messages[chat_id].copy()
        self.pending_messages[chat_id] = []

        combined_message = " ".join([msg["text"] for msg in messages]).strip()
        user_name = messages[0].get("user_name", "")

        # Simula digitação natural
        time.sleep(2)
        self.waha.start_typing(chat_id)
        time.sleep(2)

        try:
            history_messages = self.get_clean_history(chat_id)
            response_message = self.ai_bot.invoke(
                history_messages=history_messages,
                question=combined_message,
                user_name=user_name,
            )
            self.waha.send_message(chat_id, response_message)
        except Exception as e:
            print(f"[ERRO] Falha ao processar mensagem: {str(e)}")
        finally:
            self.waha.stop_typing(chat_id)

    def get_clean_history(self, chat_id):
        """Obtém e limpa o histórico de mensagens de forma robusta"""
        if chat_id not in self.message_histories:
            self.message_histories[chat_id] = []

        # Recupera o histórico recente
        raw_history = self.waha.get_history_messages(chat_id, limit=10)
        self.message_histories[chat_id].extend(raw_history)

        filtered_history = [
            msg
            for msg in self.message_histories[chat_id]
            if msg.get("body", "").strip()
        ]
        clean_history = self.remove_duplicates(filtered_history)

        self.message_histories[chat_id] = clean_history
        return clean_history

    def remove_duplicates(self, history):
        """Remove mensagens duplicadas do histórico"""
        seen_contents = {}
        clean_history = []

        for msg in history:
            content = msg.get("body", "").strip()
            is_from_me = msg.get("fromMe", False)

            key = f"{'bot' if is_from_me else 'user'}:{content}"

            if key not in seen_contents:
                clean_history.append(msg)
                seen_contents[key] = True

        return clean_history
