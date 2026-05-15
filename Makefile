.PHONY: install up down logs test lint dev

install:
	pip install -e ".[dev]"

up:
	docker-compose up -d --build

down:
	docker-compose down

logs:
	docker-compose logs -f pagehub-llm-gateway

dev:
	uvicorn api.main:app --host 0.0.0.0 --port 4011 --reload

test:
	pytest

lint:
	ruff check api tests
