export PYTHONPATH="$PYTHONPATH:$PWD"

python src/generation_run.py --dataset data/dataset.parquet \
                            --config src/config/phi-3-mini-128k_cfg.json \
                            --db-config "src/config/vstore_naive_cfg.json" 