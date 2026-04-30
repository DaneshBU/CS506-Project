.PHONY: install run test clean

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt

run:
	. .venv/bin/activate && python3 final_code.py

test:
	. .venv/bin/activate && pytest tests/

clean:
	rm -rf .venv __pycache__ .pytest_cache
	rm -f XGBoost_test_predictions.csv xgboost_feature_importance.csv
	rm -rf .venv __pycache__ .pytest_cache tests/__pycache__ outputs

