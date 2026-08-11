#!/bin/bash
# scripts/deploy.sh
# Production deployment script

set -e

echo "🚀 Starting deployment of RAG Enterprise Platform..."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if .env.prod exists
if [ ! -f .env.prod ]; then
    echo -e "${RED}❌ .env.prod file not found!${NC}"
    echo "Please create .env.prod from .env.example"
    exit 1
fi

# Load environment variables
export $(cat .env.prod | grep -v '^#' | xargs)

# Pull latest image
echo -e "${YELLOW}📥 Pulling latest image...${NC}"
docker-compose -f docker-compose.prod.yml pull

# Stop and remove old containers
echo -e "${YELLOW}🛑 Stopping old containers...${NC}"
docker-compose -f docker-compose.prod.yml down

# Start new containers
echo -e "${YELLOW}🚀 Starting new containers...${NC}"
docker-compose -f docker-compose.prod.yml up -d

# Wait for health check
echo -e "${YELLOW}⏳ Waiting for services to be healthy...${NC}"
sleep 10

# Check health
echo -e "${YELLOW}🔍 Checking health...${NC}"
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)

if [ $HEALTH_STATUS -eq 200 ]; then
    echo -e "${GREEN}✅ Deployment successful!${NC}"
else
    echo -e "${RED}❌ Health check failed!${NC}"
    echo "Check logs: docker-compose -f docker-compose.prod.yml logs"
    exit 1
fi

# Show status
docker-compose -f docker-compose.prod.yml ps

echo -e "${GREEN}🎉 Deployment complete!${NC}"
echo "📍 API: http://localhost:8000"
echo "📍 Docs: http://localhost:8000/docs"
echo "📍 Grafana: http://localhost:3000"