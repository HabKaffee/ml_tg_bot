#!/bin/bash

source config/bootstrap.sh

isort config/ --check
isort src/ --check
