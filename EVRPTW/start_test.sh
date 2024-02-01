#!/bin/bash

export PYTHONPATH="${PWD}"


pytest "${PWD}/tests/test_target.py" -v
pytest "${PWD}/tests/test_evrptw_config.py" -v
pytest "${PWD}/tests/test_ant.py" -v
pytest "${PWD}/tests/test_acsd.py" -v


