import bitsandbytes as bnb
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


def load_model(model_id:str) -> AutoModelForCausalLM:
    """
    Load a model.

    Args:
        model_id (str): Model identifier.

    Returns:
        AutoModelForCausalLM: Model.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    return model


def load_quantized_model(model_id:str) -> AutoModelForCausalLM:
    """
    Load a quantized model using BitsAndBytes.

    Args:
        model_id (str): Model identifier.

    Returns:
        AutoModelForCausalLM: Quantized model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Normalized float 4 
        bnb_4bit_use_double_quant=True,  # Double quantization for better efficincy
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype="auto",
        device_map="auto",
        quantization_config=bnb_config,
    )
    return model


def load_model_tokenizer(model_id:str, quant=False) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer.

    Args:
        model_id (str): Model identifier.

    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: Model and tokenizer.
    """

    if quant:
        model = load_quantized_model(model_id)
    else:
        model = load_model(model_id)
           
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # In case pad_token is not set
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer 