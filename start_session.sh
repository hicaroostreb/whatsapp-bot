#!/bin/bash

WAHA_URL=http://wpp_bot_waha:3000
SESSION_NAME=default
CONFIG_PATH=/app/session_config.json
EXPECTED_WEBHOOK="http://api:8000/chatbot/webhook/"

# Aguardar o Waha estar no ar
echo "⏳ Aguardando Waha..."
until curl -s "$WAHA_URL/health" > /dev/null; do sleep 2; done

# Criar a sessão
echo "✅ Waha está no ar. Criando sessão..."
curl -s -X POST -H "Content-Type: application/json" --data-binary @"$CONFIG_PATH" "$WAHA_URL/api/sessions"

# Verificar webhook
echo "🔍 Verificando webhook..."
webhook_url=$(curl -s "$WAHA_URL/api/sessions/$SESSION_NAME" | jq -r '.config.webhooks[0].url')

if [[ "$webhook_url" != "$EXPECTED_WEBHOOK" ]]; then
  echo "❌ Webhook errado: $webhook_url"
  exit 1
fi

# Iniciar a sessão
echo "🚀 Iniciando sessão..."
curl -s -X POST "$WAHA_URL/api/sessions/$SESSION_NAME/start"

echo "✅ Sessão iniciada!"
