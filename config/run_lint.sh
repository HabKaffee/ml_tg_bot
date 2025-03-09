#!/bin/bash

source config/bootstrap.sh

python config/py_init_checker.py --directory .

pylint config/ --rcfile config/.pylintrc
pylint src/ --rcfile config/.pylintrc
