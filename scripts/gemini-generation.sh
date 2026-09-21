export PYTHONPATH="$PYTHONPATH:$PWD"

python src/generation_run.py \
    --dataset "data/dataset.parquet" \
    --config "src/config/gemini_2_flash_cfg.json" \
    --db-config "src/config/vstore_naive_cfg.json" \
    --inference-mode "google"