import glob
import os
import json
import random

import pandas as pd

from typing import Generator
from pydantic import ValidationError
from rich import print as rprint

from dataset.file_validation import QuestionsListModel
from config.constants import WORKDIR


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize to list of docs
    df['Support Docs'] = df['Support Docs'].apply(lambda x: [x] if isinstance(x, str) else x)
    df['Support text'] = df['Support text'].apply(lambda x: [x] if isinstance(x, str) else x)
    # Normalize to string for contact email
    df['Contact email'] = df['Contact email'].apply(lambda x: x[0] if isinstance(x, list) else x)
    return df


def validate_json(data: dict):
    try:
        QuestionsListModel.model_validate(data)
    except ValidationError as e:
        with open('validation_errors.log', 'w', encoding='utf-8') as log_file:
            log_file.write(f"Validation error: {e}\n")
            rprint(f"[red]File validation errors: More info in 'validation_errors.log'.[/red]")
        raise e


def json_pandas(file_path:str) -> pd.DataFrame:
    # Leer el JSON
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  
    
    #validate_json(data)

    # Convertir el JSON en un DataFrame
    df = pd.json_normalize(data)
    df = data_cleaning(df)
    return df


def load_data(folder_path:str) -> pd.DataFrame:
     # Check if relative or absolute path
    if not os.path.isabs(folder_path):
        folder_path = os.path.join(WORKDIR, folder_path)

    files = glob.glob(folder_path + '/**/*.json', recursive=True)
    dataset = [json_pandas(file_path) for file_path in files]

    if not dataset: raise Exception("No JSON files found in the specified directory.")

    dataset = pd.concat(dataset, ignore_index=True)
    return dataset


def read_markdown(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def load_data_md(folder_path: str, random_sample: float = 1)-> Generator[str, None, None]:
    # Check if relative or absolute path
    if not os.path.isabs(folder_path):
        folder_path = os.path.join(WORKDIR, folder_path)

    files = glob.glob(folder_path + '/**/*.md', recursive=True)
    # Random sample of files
    if random_sample < 1:
        files = random.sample(files, int(len(files) * random_sample))

    if not files:
        raise Exception("No files found in the specified directory.")

    for file_path in files:
        yield read_markdown(file_path)
    

def md_from_path(folder_path: str, random_sample: float = 1)-> Generator[str, None, None]:
    # Check if relative or absolute path
    if not os.path.isabs(folder_path):
        folder_path = os.path.join(WORKDIR, folder_path)

    files = glob.glob(folder_path + '/**/*.md', recursive=True)
    # Random sample of files
    if random_sample < 1:
        files = random.sample(files, int(len(files) * random_sample))

    if not files:
        raise Exception("No files found in the specified directory.")

    return files


def n_files(folder_path: str, random_sample:float = 1) -> int:
    # Check if relative or absolute path
    if not os.path.isabs(folder_path):
        folder_path = os.path.join(WORKDIR, folder_path)

    files = glob.glob(folder_path + '/**/*.md', recursive=True)
    if random_sample < 1:
        files = random.sample(files, int(len(files) * random_sample))
    return len(files)


def extract_noise_url(data: pd.DataFrame) -> list[str]:
    urls = set()
    for element in data.iterrows():
        for i in range(1, 11):
            if (
                element[1][f'Search results.pos_{i}.url'] and 
                not element[1][f'Search results.pos_{i}.Correct info?'] and 
                not element[1][f'Search results.pos_{i}.misleading info']
            ):
                urls.add(element[1][f'Search results.pos_{i}.url'])
    
    return list(urls)
