TAG?=0.1.0
registry=keti.asuscomm.com:5001
current_dir=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

all: test

tests:
	py.test -n 28 --cov-report term-missing --cov XNOR-Net tests/	