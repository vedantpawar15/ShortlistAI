PYTHON ?= python

.PHONY: install test lint format api dashboard clean

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

api:
	$(PYTHON) -m uvicorn src.api:app --host 0.0.0.0 --port 8000

dashboard:
	$(PYTHON) -m streamlit run app.py

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"

