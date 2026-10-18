.PHONY: install run debug clean lint lint-strict
FLAKE8_EXCLUDE = .venv,venv,.git,__pycache__,.mypy_cache,.pytest_cache,llm_sdk
MYPY_EXCLUDE = (\.venv|llm_sdk)

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf data/output

fclean: clean
	rm -rf .venv 

lint:
	uv run flake8 . --exclude=$(FLAKE8_EXCLUDE)
	uv run mypy . --exclude '$(MYPY_EXCLUDE)' --follow-imports=silent --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . --exclude=$(FLAKE8_EXCLUDE)
	uv run mypy . --exclude '$(MYPY_EXCLUDE)' --follow-imports=silent --strict