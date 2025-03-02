#!/bin/bash

source config/bootstrap.sh

black config/ --check
black src/ --check
