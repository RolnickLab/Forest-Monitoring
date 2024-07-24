# Makefile for project setup

.PHONY: all setup download install install-dev

all: setup

setup: download install install-dev

download:
	@echo "Downloading data..."
	python data_download.sh

install:
	@echo "Installing required packages..."
	pip install -r requirements.txt

install-dev:
	@echo "Installing the treemonitoring package"
	pip install -e .

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete