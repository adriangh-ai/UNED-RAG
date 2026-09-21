import argparse
import json
import os
import gc

import torch
import pandas as pd

from datetime import datetime
from tqdm import tqdm
from transformers import pipeline
from rich.console import Console

tqdm.pandas()

from inference.context_templates import ContextConfig, BaseContextConfig
from inference.load_model import load_model_tokenizer
from vector_store.database import (
    vector_store_init, 
    vector_store_init_late_emb, 
    create_ensemble_retriever
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a specified model and dataset.")
    parser.add_argument('--model_id', type=str, help="The ID of the HF model")
    parser.add_argument('--dataset', type=str, help="The dataset path")
    parser.add_argument('--config', type=str, help="Path to the JSON configuration file")
    parser.add_argument('--db-config', type=str, help="Path to the JSON database configuration file")
    parser.add_argument('--system', type=str, default='base', help="The system identifier: base, naive, late_chunk")
    parser.add_argument('--api', type=str, default='hf', help="The API to use: openai, hf, google")
    return parser.parse_args()


def extract_after_think(text):
    return text.split('</think>\n\n')[-1]


def main(system, model_id, dataset, config, db_config):
    # Load the dataset
    logger = Console()
    logger.log(f"Loading dataset from {dataset}", style="bold blue")

    df = pd.read_parquet(dataset)

    # Additional generation parameters. HuggingFace defaults
    generation_params = {
        'max_new_tokens': config.get('max_new_tokens', 512),
        'do_sample': config.get('do_sample', False),
        'temperature': config.get('temperature', 1e-5),
        'top_k': config.get('top_k', 50),
        'top_p': config.get('top_p', 0.95),
        'num_return_sequences': config.get('num_return_sequences', 1),
        'batch_size': config.get('batch_size', 8),
        'use_cache': config.get('use_cache', True),
        'num_beams': config.get('num_beams', 1),
        'length_penalty': config.get('length_penalty', 1.0),
        'early_stopping': config.get('early_stopping', False),
        'use_cache': True
    }

    k = db_config.get('k', 10)

    match system:
        case 'base':
            logger.log("Generating base context...", style="bold blue")
            df['context'] = df.progress_apply(
                lambda x: BaseContextConfig.generate_base_context(text=x['Question']),
                axis=1
            )

        case 'naive':
            logger.log("Loading vector store...", style="bold blue")
            # Initialize the vector store
            vector_store = vector_store_init(**db_config)
            
            logger.log("Generating naive context...", style="bold blue")
            # Generate the context
            df['rag_output'] = df['Question'].progress_apply(
                lambda x: vector_store.similarity_search(x, k=k)
            )
            df['rag_output'] = df['rag_output'].apply(
                lambda x: '\n'.join([doc.page_content for doc in x])
            )
            df['context'] = df.progress_apply(
                lambda x: ContextConfig.generate_context(text=x['Question'], context=x['rag_output']),
                axis=1
            )
            del vector_store
            gc.collect()
            torch.cuda.empty_cache()

        case _:
            raise ValueError(f"System {system} not recognized.")
    
    logger.log("Loading generative model for text-generation", style="bold blue")
    # Load the model, tokenizer and hf pipeline
    model, tokenizer = load_model_tokenizer(model_id, device=config.get('device', 'auto'))
    generator = pipeline(
        'text-generation', 
        model=model, 
        tokenizer=tokenizer,
        framework=config.get('framework', 'pt'),
    )

    logger.log("Running inference...", style="bold blue")
    # Run inference
    df['output'] = df['context'].progress_apply(
        lambda x: extract_after_think(
            generator(x, return_full_text=False, **generation_params)[-1]['generated_text']
        )
    )

    # Create the results directory
    model_name = model_id.split('/')[-1]
    identifier = db_config.get('identifier', 'default')
    date_str = datetime.now().strftime('%Y%m%d')
    results_dir = os.path.join('results', f"{system}_{model_name}_{identifier}_{date_str}")
    os.makedirs(results_dir, exist_ok=True)

    # Save the config and db_config files
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    with open(os.path.join(results_dir, 'db_config.json'), 'w') as f:
        json.dump(db_config, f, indent=4)

    # Save the results dataframe
    df.to_parquet(os.path.join(results_dir, 'results.parquet'))

    logger.log(f"Results saved in {results_dir}", style="bold green")


if __name__ == "__main__":
    args = parse_args()

    model_id = args.model_id
    dataset = args.dataset
    system = args.system

    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        model_id = config.get('model_id', model_id)
        dataset = config.get('dataset', dataset)

    db_config = {}
    if args.db_config:
        with open(args.db_config, 'r') as f:
            db_config = json.load(f)
    
    if not model_id or not dataset:
        raise ValueError(
            "Model ID and dataset must be provided either as arguments or in the config file."
        )

    main(system, model_id, dataset, config, db_config)