import argparse
import json
import os
import re
import time

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from openai import OpenAI
from inference.api_generation import openai_request
from inference import context_templates

from config.constants import WORKDIR


def extract_scores(text):
    """
    Extracts evaluation scores from a text containing a JSON block.

    Args:
        text (str): The text containing the JSON block with evaluation scores.

    Returns:
        dict: A dictionary with evaluation scores if extraction is successful, None otherwise.
    """
    try:
        # Extraer posible bloque JSON con regex robusta
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if not match:
            raise ValueError("JSON block not found in the text.")
        
        json_block = match.group(0)

        scores = json.loads(json_block)

        expected_keys = {"veracidad", "precision", "consistencia", "fuentes", "objetividad"}
        if not expected_keys.issubset(scores.keys()):
            raise ValueError(f"Thera are keys missing: {expected_keys - set(scores.keys())}")

        return scores
    
    except Exception as e:
        print("Annotation error", e)
        return None


def main(dataset:pd.DataFrame, method:str):
        cgpt = os.getenv('OPENAI_API_KEY')
        client = OpenAI(api_key=cgpt)
        generator = openai_request
        generation_params = {
            'client': client,
        }

        results_dir = os.path.join(WORKDIR, 'results_llm', 'openai')
        # Create the results directory
        os.makedirs(results_dir, exist_ok=True)

        # Load dataset
        suffix = method + "_" + time.strftime("%Y-%m-%d_%H-%M-%S")
        df = pd.read_parquet(dataset)

        if 'rag_output' not in df.columns:
            df['rag_output'] = 'No se ha provisto contexto adicional.'
        # Run inference
        df['llm_eval_raw'] = df.progress_apply(
            lambda x: generator(
                context_templates.LlmJudgeConfig.generate_context(
                    question=x['Question'],
                    context=x['rag_output'],
                    output=x['output']
                ),
                client
            )[-1]['generated_text'],
            axis=1
        )

        # Extract scores
        df['llm_eval'] = df['llm_eval_raw'].progress_apply(lambda x: extract_scores(x))
        # Save results
        results = df[['id', 'llm_eval_raw', 'llm_eval']].to_dict(orient='records')
        with open(os.path.join(results_dir, f'results_{suffix}.json'), 'w') as f:
            json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a generative model.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset to use.")

    args = parser.parse_args()
    dataset = args.dataset

    method = ""
    if "late_chunk" in dataset.lower():
        method = "late_chunk"
    elif "base" in dataset.lower():
        method = "base"
    elif "naive" in dataset.lower():
        method = "naive"
    main(dataset, method)