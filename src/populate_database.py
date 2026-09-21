import argparse
import json

import pandas as pd

from rich.console import Console
from tqdm import tqdm

from vector_store.database import (
    vector_store_init, 
    store_documents, 
)
from dataset.load_data import md_from_path, extract_noise_url


def parse_args():
    parser = argparse.ArgumentParser(description="Populate the database with documents.")
    parser.add_argument("--config", type=str, help="Path to JSON config file.")
    parser.add_argument("--identifier", type=str, default="datastore_naive", help="Identifier for the datastore.")
    parser.add_argument("--chunk_size", type=int, default=128, help="Size of text chunks.")
    parser.add_argument("--chunk_overlap", type=int, default=32, help="Overlap size of text chunks.")
    parser.add_argument("--model_id", type=str, default="jinaai/jina-embeddings-v3", help="Model ID.")
    parser.add_argument("--model_kwargs", type=json.loads, default='{"device": "cuda"}', help="Model kwargs as JSON string.")
    parser.add_argument("--persist_directory", type=str, default="datastore_naive_db", help="Directory to persist the datastore.")
    parser.add_argument("--data_file", type=str, default="data/dataset.parquet", help="Path to the data file.")
    return parser.parse_args()


def main():
    console = Console()

    args = parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "identifier": args.identifier,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "model_id": args.model_id,
            "model_kwargs": args.model_kwargs,
            "persist_directory": args.persist_directory
        }

    console.log(f"Loading file {args.data_file}", style="bold blue")
    data_file = config.get('data_file', args.data_file)
    
    data = pd.read_parquet(data_file)
    noise_data = extract_noise_url(data)
  
    # Extract the parameters to pass explicitly
    identifier = config.pop('identifier', "odesia_rag")
    persist_directory = config.pop('persist_directory', "./odesia_rag_db")
    config.pop('k', 10)
    
    # Initialize the vector store
    console.log("Initializing naive vector store...", style="bold blue")
    vector_db = vector_store_init(
        identifier=identifier,
        persist_directory=persist_directory,
        **config
    )

    # Store support
    console.log("Storing noise documents...", style="bold blue")
    store_documents(vector_store=vector_db, data=noise_data, **config)
 

    console.log(
        f"Documents stored in datastore '{identifier}' in {persist_directory}", 
        style="bold green"
    )


if __name__ == "__main__":
    main()