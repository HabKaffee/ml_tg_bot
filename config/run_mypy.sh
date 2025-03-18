#!/bin/bash

source config/bootstrap.sh

mypy --install-types

mypy config/
mypy src/
mypy app.py
