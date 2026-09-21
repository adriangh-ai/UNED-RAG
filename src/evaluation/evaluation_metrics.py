from evaluate import load
from pandas import DataFrame
from typing import Union


def bertscore(preds:Union[DataFrame, list], refs:Union[DataFrame, list]) -> dict:
    """
    BERTScore
    """
    bertscore = load('bertscore')

    results = bertscore.compute(
        predictions=preds,
        references=refs,
        lang="es",
        rescale_with_baseline=True,
        model_type="xlm-roberta-large",
    )
    return results


def bleu(preds:Union[DataFrame, list], refs:Union[DataFrame, list]) -> dict:
    """
    BLEU
    """
    bleu = load('bleu')
    results = bleu.compute(predictions=preds, references=refs)
    return results


def meteor(preds:Union[DataFrame, list], refs:Union[DataFrame, list]) -> dict:
    """
    METEOR
    """
    meteor = load('meteor')
    results = meteor.compute(predictions=preds, references=refs)
    return results      


def rouge(preds:Union[DataFrame, list], refs:Union[DataFrame, list]) -> dict: 
    """
    ROUGE
    """
    rouge = load('rouge')
    results = rouge.compute(predictions=preds, references=refs)
    return results


def exact_match(preds:Union[DataFrame, list], refs:Union[DataFrame, list]) -> dict:
    """
    Exact Match
    """
    exact_match = load('exact_match')
    results = exact_match.compute(predictions=preds, references=refs)
    return results