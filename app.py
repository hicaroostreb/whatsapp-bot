import time
import random
from threading import Timer
from flask import Flask, request, jsonify
from bot.ai_bot import AIBot
from services.waha import Waha

app = Flask(__name__)

# Inicializações de configurações e variáveis globais
app.processed_messages = (
    set()
)  # Conjunto de IDs já processados (mais eficiente que dict)
app.pending_messages = {}  # Armazena mensagens aguardando processamento
app.message_timers = {}  # Controla timers para cada chat
app.message_histories = {}  # Armazena históricos de mensagens para cada chat (corrigido)
WAIT_TIME = 10  # Segundos para aguardar mais mensagens

# Instanciando objetos fora das funções para evitar instanciamento repetido
waha = Waha()
ai_bot = AIBot()


@app.route("/chatbot/webhook/", methods=["POST"])
def webhook():
    """Endpoint para receber mensagens do WhatsApp"""
    data = request.json

    # Extrai dados relevantes
    chat_id = data["payload"]["from"]
    message_id = data["payload"]["id"]
    message_text = data["payload"]["body"].strip()
    user_name = data["payload"]["_data"].get("notifyName", "")

    # Filtra mensagens indesejadas
    if not message_text or "@g.us" in chat_id or "status@broadcast" in chat_id:
        return jsonify({"status": "ignored", "reason": "filtered"}), 200

    # IMPORTANTE: Verificação robusta de duplicação
    # Usa um conjunto (set) para checagem rápida
    if message_id in app.processed_messages:
        print(f"Ignorando mensagem duplicada: {message_id}")
        return jsonify({"status": "ignored", "reason": "duplicate"}), 200

    # Marca como processada imediatamente
    app.processed_messages.add(message_id)

    # Limpa o conjunto se ficar muito grande (eficiência de memória)
    if len(app.processed_messages) > 5000:
        old_messages = list(app.processed_messages)
        app.processed_messages = set(
            old_messages[-1000:]
        )  # Mantém apenas os 1000 mais recentes

    # Adiciona à fila de pendentes
    add_to_pending(chat_id, message_text, message_id, user_name)

    # Configura timer para processamento
    setup_processing_timer(chat_id)

    return jsonify({"status": "success"}), 200


def command_zerar(chat_id):
    """Lida com o comando --zerar, que reseta o histórico do chat."""

    # Apaga o histórico de mensagens desse chat específico
    if chat_id in app.message_histories:
        app.message_histories[chat_id] = []  # Limpa o histórico armazenado

    # Retorna uma resposta informando que o histórico foi resetado
    return "Histórico de conversa zerado! Podemos começar de novo."


def add_to_pending(chat_id, message_text, message_id, user_name):
    """Adiciona mensagem à lista de pendentes"""
    if chat_id not in app.pending_messages:
        app.pending_messages[chat_id] = []

    app.pending_messages[chat_id].append(
        {
            "text": message_text,
            "id": message_id,
            "user_name": user_name,
            "timestamp": time.time(),
        }
    )


def setup_processing_timer(chat_id):
    """Configura ou reconfigura o timer para processamento"""
    # Cancela timer existente se houver
    if chat_id in app.message_timers and app.message_timers[chat_id]:
        app.message_timers[chat_id].cancel()

    # Cria novo timer
    app.message_timers[chat_id] = Timer(WAIT_TIME, process_messages, args=[chat_id])
    app.message_timers[chat_id].start()


def process_messages(chat_id):
    """Processa as mensagens pendentes após o tempo de espera"""
    if chat_id not in app.pending_messages or not app.pending_messages[chat_id]:
        return

    messages = app.pending_messages[chat_id].copy()
    app.pending_messages[chat_id] = []

    combined_message = " ".join([msg["text"] for msg in messages]).strip()
    user_name = messages[0].get("user_name", "")

    # Verifica se o comando --zerar foi usado
    if combined_message.lower() == "--zerar":
        # Limpa o histórico
        if chat_id in app.message_histories:
            app.message_histories[chat_id] = []

        # Envia uma mensagem técnica de sistema, não do bot
        waha.send_message(chat_id=chat_id, message="✅ Histórico zerado.")

        # Não continua o fluxo de IA nem salva esse comando no histórico
        return

    # Simula digitação natural
    time.sleep(random.randint(1, 3))
    waha.start_typing(chat_id=chat_id)
    time.sleep(random.randint(5, 7))

    try:
        history_messages = get_clean_history(waha, chat_id)

        response_message = ai_bot.invoke(
            history_messages=history_messages,
            question=combined_message,
            user_name=user_name,
        )

        waha.send_message(chat_id=chat_id, message=response_message)
    except Exception as e:
        print(f"[ERRO] Falha ao processar mensagem: {str(e)}")
    finally:
        waha.stop_typing(chat_id=chat_id)


def get_clean_history(waha, chat_id):
    """Obtém e limpa o histórico de mensagens de forma robusta, mantendo um histórico acumulado"""

    # Tenta recuperar o histórico armazenado do chat (se já houver)
    if chat_id not in app.message_histories:
        app.message_histories[chat_id] = []

    # Obtém o histórico recente (últimas 10 mensagens)
    raw_history = waha.get_history_messages(chat_id=chat_id, limit=10)

    # Adiciona as mensagens recentes ao histórico acumulado
    app.message_histories[chat_id].extend(raw_history)

    # Remove mensagens vazias
    filtered_history = [
        msg for msg in app.message_histories[chat_id] if msg.get("body", "").strip()
    ]

    # Remove duplicações de forma mais controlada
    clean_history = []
    seen_contents = {}

    for msg in filtered_history:
        content = msg.get("body", "").strip()
        is_from_me = msg.get("fromMe", False)

        # Cria chave única para cada tipo de mensagem, removendo variações simples
        key = f"{'bot' if is_from_me else 'user'}:{normalize(content)}"  # Normaliza o conteúdo para remover pequenas variações

        # Só adiciona se não for duplicada
        if key not in seen_contents:
            clean_history.append(msg)
            seen_contents[key] = True

    # Atualiza o histórico acumulado com a versão limpa
    app.message_histories[chat_id] = clean_history

    return clean_history


def normalize(content):
    """Normaliza o conteúdo da mensagem para remover espaços extras e pontuação desnecessária"""
    # Remover espaços extras
    content = " ".join(content.split())
    # Remover pontuação extra (se necessário)
    # content = content.translate(str.maketrans("", "", string.punctuation))
    return content


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
