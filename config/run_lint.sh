#!/bin/bash

source config/bootstrap.sh

set -e
python config/py_init_checker.py --directory .

pylint config/
pylint src/
pylint app.py
