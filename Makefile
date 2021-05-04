all: test

test:
	pytest -o log_cli=true -n 28 --cov-report term-missing --cov XNOR-Net tests/	