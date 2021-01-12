all: test

test:
	py.test -n 28 --cov-report term-missing --cov XNOR-Net tests/	