VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip
FLAKE8 = $(VENV)/bin/flake8
MYPY = $(VENV)/bin/mypy
EXCLUDE = venv,.mypy_cache,.pytest_cache,__pycache__

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: install
	$(PYTHON) main.py

debug:
	$(PYTHON) -m pdb main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	
fclean: clean
	rm -rf $(VENV)

lint:
	$(FLAKE8) . --exclude=$(EXCLUDE)
	$(MYPY) . --exclude=venv --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(FLAKE8) . --exclude=$(EXCLUDE)
	$(MYPY) . --exclude=venv --strict

.PHONY: install run debug clean lint lint-strict