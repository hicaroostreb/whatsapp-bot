# Vexus Chatbot 🤖

An intelligent chatbot for automated customer service via WhatsApp, leveraging Gemini AI for contextual responses and offering persistent conversation history.

---

## ✨ Features

-   **Gemini AI Integration:** Contextual responses powered by `AIBot`.
-   **WhatsApp Connection:** Seamless integration using the `Waha` library.
-   **Persistent History:** Maintains conversation history for contextual AI interactions.
-   **`--zerar` Command:** Resets chat history with a hidden command.
-   **Spam Filter:** Robust protection against duplicate messages and spam.
-   **Intelligent Timer:** Groups messages by session for better context.
-   **Docker Ready:** Easy deployment with `docker-compose`.

---

## 🐳 Running with Docker

> Ensure you have [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed.

### Steps

1.  Clone the repository:

    ```bash
    git clone https://github.com/seu-usuario/vexus-chatbot.git
    cd vexus-chatbot
    ```

2.  Start the services:

    ```bash
    docker-compose up --build api
    ```

    The Flask backend will be accessible at: `http://localhost:8000`

    Other services: 

    -   Webhook: `http://api:8000/chatbot/webhook/`
    -   WAHA API Swagger: `http://localhost:3000/`
    -   WAHA Dashboard: `http://localhost:3000/dashboard`

---

## 📡 Webhook

The chatbot listens for messages via a webhook:

```
POST /chatbot/webhook/
Content-Type: application/json

Payload:

```json
{
  "payload": {
    "from": "5551999999999@c.us",
    "id": "ABCD123456",
    "body": "Olá, quero saber mais sobre consórcios!",
    "_data": {
      "notifyName": "João Cliente"
    }
  }
}
```

---

## 🧠 Special Command: `--zerar`

Clears the chat history when a user sends `--zerar`. The bot returns:

```
✅ Histórico zerado.
```

This command is not stored and prevents automatic AI responses.

---

## 🗂 Folder Structure

```
vexus-chatbot/
├── bot/
│   └── ai_bot.py         # AIBot class, interface with Gemini AI
├── services/
│   └── waha.py           # Service for sending messages via WhatsApp
├── app.py                # Main application entry point (Flask)
├── docker-compose.yml    # Container orchestration
├── Dockerfile.api        # Container definition for the API
├── README.md             # This file
└── requirements.txt      # Project dependencies
```

---

## 🧪 Tests

In development.

---

## 📃 License

MIT © 2025 — [Seu Nome ou Organização]
