from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

from config import settings


def load_llama():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LlamaCpp(
        model_path=settings.MODEL_PATH,
        temperature=settings.TEMPERATUTE,
        max_tokens=settings.MAX_TOKENS,
        top_p=settings.TOP_P,
        n_gpu_layers=settings.N_GPU_LAYERS,
        callback_manager=callback_manager,
        repeat_penalty=settings.REPEAT_PENALTY,
        f16_kv=settings.F16_KV,
        verbose=settings.LLAMA_VERBOSE,
        n_ctx=2048
    )
    return llm
