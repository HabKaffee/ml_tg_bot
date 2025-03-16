#!/bin/bash

source config/bootstrap.sh

set -E
python config/py_init_checker.py --directory .

pylint config/
pylint src/
