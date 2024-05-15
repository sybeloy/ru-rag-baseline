import os
import logging

from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import TFIDFRetriever

from config import settings
from data import load_documents

logging.basicConfig(level=logging.DEBUG, filename=settings.LOG_FILEPATH)


class TfIdfRetriever:
    def load(
        self,
        data_path: str = settings.DATA_PATH,
        force_recompute: bool = settings.RECOMPUTE_ON_APP_RELOAD,
    ) -> BaseRetriever:
        """Load retriever

        Args:
            data_path (str, optional): Path to data for RAG.
            force_recompute (bool, optional): Defaults to settings.RECOMPUTE_ON_APP_RELOAD.

        Returns:
            BaseRetriever
        """
        if force_recompute or not os.path.exists(settings.TFIDF_INDEX_SAVE_PATH):
            logging.debug("Computing TF-IDF index...")

            retriever = TFIDFRetriever.from_documents(load_documents(data_path))
            retriever.save_local(settings.TFIDF_INDEX_SAVE_PATH)
            return retriever
        else:
            logging.debug("Loading TF-IDF index from disk")

        return TFIDFRetriever.load_local(
            settings.TFIDF_INDEX_SAVE_PATH, allow_dangerous_deserialization=True
        )


class RerankRetriever:
    def __init__(self) -> None:
        """Retriever with cross-encoder reranking"""
        self.encoder = HuggingFaceCrossEncoder(model_name=settings.CROSS_ENCODER_MODEL)

    def load(self, retriever: BaseRetriever) -> BaseRetriever:
        """Load retriever

        Args:
            retriever (BaseRetriever)

        Returns:
            BaseRetriever
        """
        compressor = CrossEncoderReranker(
            model=self.encoder, top_n=settings.RERANKER_TOP_N
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever,
        )
        return compression_retriever
