#!/bin/bash

source config/bootstrap.sh

set -e

python config/py_init_checker.py --directory config/
pylint config/

python config/py_init_checker.py --directory src/
pylint src/

pylint app.py
