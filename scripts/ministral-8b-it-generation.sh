export PYTHONPATH="$PYTHONPATH:$PWD"

python src/generation_run.py --dataset data/dataset.parquet \
                            --config src/config/ministral-8b-it-2410_cfg.json \
                            --db-config "src/config/vstore_naive_cfg.json" \