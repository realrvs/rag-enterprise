#!/bin/bash
# scripts/prod-setup.sh
# First-time production setup

set -e

echo "🚀 Setting up RAG Enterprise Platform for production..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. Create directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p data/raw data/processed
mkdir -p nginx/conf.d
mkdir -p grafana/provisioning
mkdir -p logs

# 2. Copy example environment
if [ ! -f .env.prod ]; then
    echo -e "${YELLOW}📝 Creating .env.prod...${NC}"
    cp .env.example .env.prod
    echo -e "${RED}⚠️ Please edit .env.prod with your configuration!${NC}"
fi

# 3. Build and start services
echo -e "${YELLOW}🐳 Building and starting services...${NC}"
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# 4. Wait for services
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 15

# 5. Pull LLM model
echo -e "${YELLOW}📥 Pulling LLM model...${NC}"
./scripts/prod-pull-model.sh qwen2.5-coder:7b

# 6. Index documents
echo -e "${YELLOW}📄 Indexing documents...${NC}"
docker exec rag-prod-app python scripts/index_documents.py

# 7. Check health
echo -e "${YELLOW}🔍 Checking health...${NC}"
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)

if [ $HEALTH_STATUS -eq 200 ]; then
    echo -e "${GREEN}✅ Setup complete!${NC}"
    echo ""
    echo "📍 API: http://localhost:8000"
    echo "📍 Docs: http://localhost:8000/docs"
    echo "📍 Grafana: http://localhost:3000 (admin/your_password)"
    echo "📍 Prometheus: http://localhost:9090"
    echo "📍 Jaeger: http://localhost:16686"
else
    echo -e "${RED}❌ Health check failed!${NC}"
    echo "Check logs: docker-compose -f docker-compose.prod.yml logs"
fi