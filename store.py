import os
import logging

from tqdm import tqdm

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings
from data import load_documents

logging.basicConfig(level=logging.DEBUG, filename=settings.LOG_FILEPATH)


class FaissVectorStore:
    def __init__(self) -> None:
        """FAISS vector store class with form index progress bar"""
        self.store = FAISS
        self.encoder = HuggingFaceEmbeddings(model_name=settings.BI_ENCODER_MODEL)

    def load(
        self,
        data_path: str = settings.DATA_PATH,
        force_recompute: bool = settings.RECOMPUTE_ON_APP_RELOAD,
    ) -> VectorStore:
        """Loading the vector storage method. If RECOMPUTE_ON_APP_RELOAD = True, recalculation
          index every time the application is launched.
          Otherwise, calculate the index the first time you load it, save it, and load the index after that.
          from disk.

        Args:
            data_path (str, optional): Path to data for RAG.
            force_recompute (bool, optional): Defaults to settings.RECOMPUTE_ON_APP_RELOAD.

        Returns:
            VectorStore
        """
        if force_recompute or not os.path.exists(settings.FAISS_INDEX_SAVE_PATH):
            logging.debug("Computing FAISS index...")
            documents = load_documents(data_path)
            store = self.form_index(documents)
            store.save_local(settings.FAISS_INDEX_SAVE_PATH)
            return store
        else:
            logging.debug("Loading FAISS index from disk")

        return self.store.load_local(
            settings.FAISS_INDEX_SAVE_PATH,
            self.encoder,
            allow_dangerous_deserialization=True,
        )

    def form_index(
        self,
        documents: list,
        encoder: Embeddings = None,
        verbose: bool = settings.SHOW_INDEX_CREATION_PROGRESS,
    ) -> VectorStore:
        """Form FAISS index method

        Args:
            documents (list): List of chunked documents
            encoder (Embeddings, optional): Encoder producing vectors for the store.
            verbose (bool, optional): Flag responsible for displaying the index formation process. If True may be less effective

        Returns:
            VectorStore: formed vector store
        """
        if not encoder:
            encoder = self.encoder
        if not verbose:
            return self.store.from_documents(documents, encoder)
        store = None
        with tqdm(total=len(documents), desc="Indexing documents") as pbar:
            for doc in documents:
                if store:
                    store.add_documents([doc])
                else:
                    store = self.store.from_documents([doc], encoder)
                pbar.update(1)
        return store
