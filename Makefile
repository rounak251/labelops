.PHONY: install test lint benchmark demo api

install:
	python -m pip install -e '.[dev]'

test:
	python -m pytest

lint:
	ruff check .
	ruff format --check .

benchmark:
	labelops benchmark --output reports

demo:
	streamlit run app.py

api:
	uvicorn labelops.api:app --host 127.0.0.1 --port 8000
