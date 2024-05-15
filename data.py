from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


def load_documents(data_path: str) -> list:
    """Data preprocessing function. Load and split data on chunks.

    Args:
        data_path (str): Path to csv or txt data file

    Raises:
        ValueError: Raises if data format not supported

    Returns:
        list: list of langchain Document class objects
    """
    if data_path.endswith(".csv"):
        documents = CSVLoader(file_path=data_path).load()
    elif data_path.endswith(".txt"):
        documents = TextLoader(file_path=data_path).load()
    else:
        raise ValueError("Data format not supported yet")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    documents = text_splitter.split_documents(documents)
    return documents
