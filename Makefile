.PHONY: help up down logs rebuild shell ps test lint fmt typecheck lock sync clean
COMPOSE := docker compose --env-file .env -f infra/docker-compose.yml

help:

		@echo "up 			- start full stack (postgres, redis, backend)"
		@echo "down 		- stop stack"
		@echo "logs 		- tail backend logs"
		@echo "rebuild		- rebuild backend images"
		@echo "shell		- open shell inside backend container"
		@echo "test 		- run pytest in container"
		@echo "lint			- ruff + mypy"
		@echo "fmt			- ruff format"
		#echo "lint-check	- ruff cli format"
		@echo "migrate		- alembic	upgrade head"
		@echo "revision		- alembic revision --autogenerate -m '...'"

up:		
		$(COMPOSE) up -d 

down:
		$(COMPOSE) down 

clean:
		$(COMPOSE) down -v 

logs: 	
		$(COMPOSE)  logs -f backend

rebuild: 	
		$(COMPOSE)  build backend

shell:
		$(COMPOSE)  exec backend bash 

test:
		cd backend && uv run pytest -v 

ps:
		$(COMPOSE) ps

lint:
		cd backend && uv run ruff check . --fix && uv run ruff format . && uv run mypy app

lint-check:
		cd backend && uv run ruff check . && uv run mypy app

fmt:
		cd backend && uv run ruff format .

