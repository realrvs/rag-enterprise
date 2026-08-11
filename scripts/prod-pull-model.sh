#!/bin/bash
# scripts/prod-pull-model.sh
# Pull LLM model for production

set -e

echo "📥 Pulling LLM model for production..."

# Check if ollama container is running
if ! docker ps | grep -q rag-prod-ollama; then
    echo "❌ Ollama container is not running!"
    echo "Run: docker-compose -f docker-compose.prod.yml up -d ollama"
    exit 1
fi

# Pull model
MODEL=${1:-qwen2.5-coder:7b}
echo "🔄 Pulling model: $MODEL"

docker exec rag-prod-ollama ollama pull $MODEL

echo "✅ Model $MODEL pulled successfully!"
echo "📊 Available models:"
docker exec rag-prod-ollama ollama list