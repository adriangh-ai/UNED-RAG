#!/bin/bash
export PYTHONPATH="$PYTHONPATH:$PWD"

python src/generate_dataset.py --file_path data/raw_data