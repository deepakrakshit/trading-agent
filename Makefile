PYTHON ?= .venv/bin/python

.PHONY: setup test doctor demo train evaluate account

setup:
	bash scripts/setup.sh

test:
	$(PYTHON) -m pytest -q
	$(PYTHON) -m compileall -q src tests
	$(PYTHON) -m pip check

doctor:
	$(PYTHON) -m trading_agent.cli doctor

demo:
	$(PYTHON) -m trading_agent.cli demo

train:
	$(PYTHON) -m trading_agent.cli train

evaluate:
	$(PYTHON) -m trading_agent.cli evaluate

account:
	$(PYTHON) -m trading_agent.cli account --db runs/demo/account.sqlite
