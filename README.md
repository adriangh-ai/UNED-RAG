# UNED-RAG

Code, data, and reproduction scripts for the study included in [94_UNED_RAG_ES_A_Spanish_Datas.pdf](94_UNED_RAG_ES_A_Spanish_Datas.pdf). The project builds a UNED question set, creates a vector store with distractor documentation, and runs generative models to study responses with retrieved context.

## About UNED-RAG ES

UNED-RAG ES is a manually curated Spanish resource for evaluating the **generation** component of Retrieval-Augmented Generation (RAG) independently from retrieval quality. It focuses on academic and administrative information needs at a Spanish university and uses publicly available UNED web content.

The dataset contains 261 canonical test units. Each unit includes:

- A natural-language question in Spanish.
- Two paraphrased reference answers with the same factual content.
- One or more supporting documents and explicitly marked minimal evidence passages.
- Up to ten semantically related but non-supporting documents, used as realistic distractor candidates.

Questions are annotated as either **Localize**, where a single supporting passage is sufficient, or **Relate/Reason**, where evidence must be combined or lightly reasoned over. This design keeps the supporting evidence traceable while allowing controlled construction of noisy contexts.

## Benchmark Design

The paper derives two benchmark families from the canonical units:

- **Noise quantity:** compares no-context and support-only baselines with contexts containing noise at 1, 2, 3, or 4 times the length of the supporting evidence, plus an all-noise setting without the support evidence.
- **Evidence position:** holds noise fixed at four times the support length and places the supporting evidence first, in the middle, or last in the prompt.

For the noise conditions, distractor documents are converted to text, split into 256-token chunks with no overlap, embedded with `jinaai/jina-embeddings-v3`, and indexed in a vector database. At evaluation time, the most similar noise chunks are retrieved for each question while excluding its supporting sources.

## Main Findings

Across BLEU, METEOR, ROUGE, and BERTScore, performance declines as semantic noise increases, with the largest drop in the all-noise setting. For several models, answering without context performs better than using only misleading context. Evidence position also affects models differently under fixed high noise. These results motivate conservative context filtering, calibrated abstention, and prompt assembly that avoids burying supporting evidence among distractors.

## Structure

- `data/dataset.parquet`: UNED-RAG dataset.
- `database/`: location of the persistent vector store (created locally).
- `src/`: implementation for dataset generation, indexing, inference, and evaluation.
- `src/config/`: model and vector-store configurations.
- `scripts/`: reproduction commands for each stage and model.
- `94_UNED_RAG_ES_A_Spanish_Datas.pdf`: paper associated with this repository.

## Requirements

- Python 3.10 or later.
- A hardware-compatible PyTorch installation to run local models. The default vector-store and local-model configuration uses CUDA.
- Access to Hugging Face to download the configured models. Some models may require accepting their terms of use.
- An API key for the applicable remote runs: OpenAI or Google Gemini.

From the repository root, create an environment and install the declared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Also install PyTorch for your platform by following [pytorch.org](https://pytorch.org/get-started/locally/), plus the direct runtime dependencies not pinned in `requirements.txt`:

```bash
python -m pip install transformers python-dotenv rich tqdm
```

Runs use the imports in `src`; launch the included scripts from the repository root.

## Reproduction Workflow

### 1. Preload the Vector Store

This step is required before generation. It extracts distractor URLs from the dataset and indexes them in Chroma with the configured embedding model. The default configuration is in `src/config/vstore_naive_cfg.json` and persists the database in `database/datastore_naive_db`.

```bash
bash scripts/load_vectorstore_naive.sh
```

To modify the embedding model, device, chunk size, number of retrieved documents, or persistence path, edit `src/config/vstore_naive_cfg.json` and run the preload again. When changing this configuration, regenerate the database so the embeddings and index remain consistent.

### 2. Run Generation

Runs load `data/dataset.parquet`, retrieve documents from the preloaded database, and save a `results.parquet` file together with the configurations used in `results/oracle_<model>_<identifier>_<timestamp>/`.

For local Hugging Face models, run one of the available scripts:

```bash
bash scripts/llama-3.1-8b-generation.sh
bash scripts/deepseek_r1_llama_8b-generation.sh
bash scripts/ministral-8b-it-generation.sh
bash scripts/phi-3-mini-generation.sh
```

You can also run the entry point directly and choose another configuration:

```bash
python src/generation_run.py \
  --dataset data/dataset.parquet \
  --config src/config/llama-3.1-8b_cfg.json \
  --db-config src/config/vstore_naive_cfg.json
```

For API providers, export the credential before running the appropriate script:

```bash
export GEMINI_API_KEY="..."
bash scripts/gemini-generation.sh

export OPENAI_API_KEY="..."
bash scripts/gpt-generation.sh
```

Each model configuration is in `src/config/`. The inference modes supported by `generation_run.py` are `hf` (the default), `google`, and `openai`.

## Evaluation

The paper reports reference-based generation metrics: BLEU, METEOR, ROUGE, and BERTScore. The metric implementations are available in `src/evaluation/evaluation_metrics.py`.

### Optional LLM-Judge Evaluation

The repository also includes an additional OpenAI-based LLM-judge evaluation. It requires `OPENAI_API_KEY`. Point it to the results file produced in the previous step:

```bash
export OPENAI_API_KEY="..."
python src/evaluate_model.py \
  --dataset results/oracle_<model>_<identifier>_<timestamp>/results.parquet
```

This evaluation saves JSON results in `results_llm/openai/`. It is a repository utility and is separate from the paper's reported reference-based evaluation.

## Configuration and Local Data

Do not include API keys in the repository. Use environment variables or a `.env` file, which Git already ignores. The `data/dataset.parquet`, `database/`, `results/`, and `results_llm/` paths are locally generated artifacts and can use substantial disk space depending on the models and documents used.

## License

This project is distributed under the terms of the [license](LICENSE) included in the repository.