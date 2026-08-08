.PHONY: help install up down restart logs shell test lint

help:
	@echo "Available commands:"
	@echo "  install     Install Python dependencies"
	@echo "  up          Start all services (docker-compose up)"
	@echo "  down        Stop all services"
	@echo "  restart     Restart all services"
	@echo "  logs        Show logs from all services"
	@echo "  shell       Open shell in app container"
	@echo "  test        Run tests"
	@echo "  lint        Run linters (ruff, black)"
	@echo "  index       Index documents from data/raw/"
	@echo "  eval        Run RAGAS evaluation"

install:
	pip install -r requirements.txt

up:
	docker-compose up -d
	@echo "✅ Services started. API: http://localhost:8000"
	@echo "   Qdrant: http://localhost:6333"
	@echo "   Redis: redis://localhost:6379"
	@echo "   Grafana: http://localhost:3000 (admin/admin)"

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

shell:
	docker-compose exec app /bin/bash

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	ruff check src/ tests/
	black src/ tests/ --check

index:
	python scripts/index_documents.py

eval:
	python scripts/run_evaluation.py

# Запуск в development режиме (с авто-перезагрузкой)
dev:
	export PYTHONPATH=. && uvicorn src.main:app --reload --host 0.0.0.0 --port 8000