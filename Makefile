all: test

test:
	pytest -o log_cli=true -n 1 --cov-report term-missing --cov XNOR-Net tests/	