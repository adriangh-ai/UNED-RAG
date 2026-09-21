#!/bin/bash
export PYTHONPATH="$PYTHONPATH:$PWD"

python src/populate_database.py --config src/config/vstore_naive_cfg.json