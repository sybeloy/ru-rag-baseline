from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATA_PATH: str
    LOG_FILEPATH: str

    # text splitter settings
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int

    # hugginface retrieve and rerank models
    BI_ENCODER_MODEL: str
    CROSS_ENCODER_MODEL: str

    # retriever settings
    SHOW_INDEX_CREATION_PROGRESS: bool
    RETRIEVER_TOP_K: int
    RERANKER_TOP_N: int

    # Faiss VectorStore and tf-idf save settings
    RECOMPUTE_ON_APP_RELOAD: bool
    FAISS_INDEX_SAVE_PATH: str
    TFIDF_INDEX_SAVE_PATH: str

    # llm settings
    MODEL_PATH: str
    TEMPERATUTE: float
    MAX_TOKENS: int
    TOP_P: float
    N_GPU_LAYERS: int
    REPEAT_PENALTY: float
    F16_KV: bool
    LLAMA_VERBOSE: bool

    class Config:
        case_sensitive = False
        env_file = ".env"


settings = Settings()
