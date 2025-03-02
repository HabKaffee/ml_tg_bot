#!/bin/bash

source config/bootstrap.sh

python config/py_init_checker.py --directory config/
pylint config/ --rcfile config/.pylintrc

python config/py_init_checker.py --directory src/
pylint src/ --rcfile config/.pylintrc
