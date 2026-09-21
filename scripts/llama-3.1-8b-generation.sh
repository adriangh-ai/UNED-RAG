export PYTHONPATH="$PYTHONPATH:$PWD"

python src/generation_run.py --dataset data/dataset.parquet \
                            --config src/config/llama-3.1-8b_cfg.json \
                            --db-config "src/config/vstore_naive_cfg.json" \