.DEFAULT_GOAL := help
COMPOSE := docker compose
DEV     := docker compose -f docker-compose.yml -f docker-compose.dev.yml

help:  ## Liệt kê target
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n",$$1,$$2}'

proto: ## Sinh stub gRPC cho core và be
	bash scripts/gen_proto.sh

up:    ## Chạy toàn bộ stack
	$(COMPOSE) up -d

dev:   ## Chạy chế độ dev (hot reload, mở port debug)
	$(DEV) up

down:  ## Dừng
	$(COMPOSE) down

logs:  ## Xem log (S=tên service)
	$(COMPOSE) logs -f $(S)

ps:    ## Trạng thái container
	$(COMPOSE) ps

migrate: ## Chạy migration
	$(COMPOSE) exec be alembic upgrade head

warm:  ## Nạp hydration cache — bước lấy lại 15ms/query
	$(COMPOSE) exec be python -m app.tools.warm_hydration_cache

bench: ## Đo latency (kỳ vọng p50<60ms, p95<150ms khi không dùng LLM)
	$(COMPOSE) exec be python -m app.tools.bench --n 500 --concurrency 4

snapshot-pull: ## Tải snapshot từ R2 (VER=v3)
	bash scripts/snapshot_pull.sh $(VER)

snapshot-swap: ## Hot swap sang snapshot mới (VER=v3)
	bash scripts/snapshot_swap.sh $(VER)

check-hw: ## Kiểm tra NVMe + AVX trước khi deploy
	bash scripts/check_hardware.sh

test:  ## Chạy test
	$(COMPOSE) exec be pytest -q && $(COMPOSE) exec searchcore pytest -q

fmt:   ## Format
	ruff format services/be/src services/core/src services/ingest/src

lint:  ## Lint
	ruff check services/be/src services/core/src services/ingest/src

.PHONY: help proto up dev down logs ps migrate warm bench snapshot-pull snapshot-swap check-hw test fmt lint
