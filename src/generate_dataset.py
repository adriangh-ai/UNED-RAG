import argparse
import os

from rich.console import Console

from config.constants import WORKDIR
from dataset.load_data import load_data


def main(file_path):
    console = Console()
    save_path = os.path.join(WORKDIR, 'data')
 
    df = load_data(file_path)
    df.to_parquet(save_path + '/dataset.parquet', index=False)
    
    console.log(
        f"Dataset saved as 'dataset.parquet' in {save_path}", 
        style="bold green"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset from folder of JSON files.")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input files')
    args = parser.parse_args()
    
    main(args.file_path)