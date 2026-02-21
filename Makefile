SHELL := /bin/bash

up:
	docker compose -f docker/docker-compose.yml --env-file .env up -d --build

down:
	docker compose -f docker/docker-compose.yml --env-file .env down -v

logs:
	docker compose -f docker/docker-compose.yml --env-file .env logs -f api

ingest:
	docker compose -f docker/docker-compose.yml --env-file .env exec -T api python -m scripts.ingest

train-intent:
	docker compose -f docker/docker-compose.yml --env-file .env exec -T api python -m scripts.train_intent

eval:
	docker compose -f docker/docker-compose.yml --env-file .env exec -T api python -m scripts.eval_retrieval

load-test:
	docker compose -f docker/docker-compose.yml --env-file .env exec -T api python -m scripts.load_test

demo:
	curl -s http://localhost:8000/health | jq .
	curl -s http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"conversation_id":"demo","message":"Привет! Нужен прайс и условия доставки."}' | jq .
	curl -s http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"conversation_id":"demo","message":"А если я хочу коммерческое предложение, куда оставить контакты?"}' | jq .