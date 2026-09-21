from typing import Generator
from markdownify import markdownify
from docling.document_converter import DocumentConverter
from langchain_text_splitters import TokenTextSplitter


# Define a function to tokenize and split the text
def split_text_into_chunks(text: str, chunk_size: int=128) -> list[str]:
    """
    Splits a given text into chunks of the specified size using the model tokenizer.

    Args:
        text (str): The input text to split.
        chunk_size (int): The number of tokens per chunk. Default is 128.

    Returns:
        List[str]: A list of text chunks.
    """
    text_splitter = TokenTextSplitter(
        encoding_name="gpt2",
        chunk_size=chunk_size,
        chunk_overlap=0  # No overlap
    )
    
    chunks = text_splitter.split_text(text)
    return chunks


def doc_to_text(document: str) -> str:
    """
    Convert a document to text.

    Args:
        document (str): The document to convert.

    Returns:
        str: The text content of the document.
    """
    converter = DocumentConverter()

    result = converter.convert(document)
    result = markdownify(result.document.export_to_markdown())
    return result


def token_text_splitter(
        text: str, 
        tokenizer, 
        chunk_size: int = 8000, 
        chunk_overlap: int = 0
    ) -> Generator[str, None, None]:
    """
    Splits a given text into chunks using the provided tokenizer.

    Args:
        text (str): The input text to split.
        tokenizer: The tokenizer to use for splitting the text.
        chunk_size (int): The number of tokens per chunk. Default is 8000.
        chunk_overlap (int): The number of tokens to overlap between chunks. Default is 0.

    Yields:
        str: A chunk of the text.
    """
    tokens = tokenizer(text)["input_ids"]
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk = tokens[i:i + chunk_size]
        yield tokenizer.decode(chunk)