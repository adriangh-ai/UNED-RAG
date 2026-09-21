import uuid

import torch
import pandas as pd

from typing import Optional, Dict, Union

from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from transformers import AutoTokenizer
from markdownify import markdownify
from docling.document_converter import DocumentConverter
from rich.console import Console

tqdm.pandas()

from vector_store.data_preprocess import split_text_into_chunks
from config.constants import WORKDIR
from dataset.load_data import read_markdown


def vector_store_init(
    model_id: Optional[str] = None,
    identifier: str = "odesia_rag",
    persist_directory: str = "./odesia_rag_db",
    **db_kwargs: Dict
) -> Chroma:
    model_id = db_kwargs.get('model_id', model_id)
    collection_name = db_kwargs.get('identifier', identifier)
    persist_directory = db_kwargs.get('persist_directory', persist_directory)

    model_kwargs = db_kwargs.get('model_kwargs', {'device': 'cuda', "trust_remote_code": True})

    hf = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs=model_kwargs,
    )

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=hf,
        persist_directory=persist_directory,
    )

    return vector_store


def add_documents(vector_store: Chroma, chunked_doc: list[str], url:str) -> None:
    doc_list = [Document(page_content=chunk, metadata={'source': url}) for chunk in chunked_doc]
    vector_store.add_documents(doc_list)


def store_documents(vector_store: Chroma, data: list[str],  **kwargs) -> None:
    # Initialize the logger
    logger = Console()

    # Extract the parameters
    chunk_size = kwargs.get('chunk_size', 256)
    chunk_overlap = kwargs.get('chunk_overlap', 0)
    model_id = kwargs.get('model_id', None)
    if model_id is None: raise ValueError("Model ID is required to store documents.")

    # Initialize the document converter and tokenizer
    doc_converter =  DocumentConverter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer, 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    # Process and store the documents
    for url in tqdm(data, desc="Storing documents"):
        try:
            if url.startswith("http"):
                _doc = doc_converter.convert(url)
                _doc = markdownify(_doc.document.export_to_markdown())
            elif url.endswith(".md"):
                url = url if url.startswith(WORKDIR) else f"{WORKDIR}/{url}"
                _doc = read_markdown(url)
            else: raise ValueError("Unsupported document format.")
        except Exception as e:
            print(f"Error processing document {url}: {e}")
            continue

        doc_chunks = text_splitter.split_text(_doc)
        if doc_chunks: add_documents(vector_store, doc_chunks, url)
        else: logger.log(f"Document {url} is empty.", style="bold yellow")


def vector_store_query(vector_store: Chroma, query: str, **kwargs):
    query = split_text_into_chunks(query, chunk_size=kwargs.get('chunk_size', 128))
    query = [Document(page_content=sample) for sample in query]
    return vector_store.query(query)