import os
import logging
import warnings

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever

from config import settings
from llama import load_llama
from store import FaissVectorStore
from retriever import RerankRetriever, TfIdfRetriever

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.DEBUG, filename=settings.LOG_FILEPATH)

if __name__ == "__main__":
    logging.info("Start application...")

    vector_store = FaissVectorStore().load()

    dense_retriever = vector_store.as_retriever(
        search_kwargs={"k": settings.RETRIEVER_TOP_K}
    )
    sparse_retriever = TfIdfRetriever().load()
    ensemble_retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, dense_retriever], weights=[0.5, 0.5]
    )

    retriever = RerankRetriever().load(ensemble_retriever)

    prompt = PromptTemplate.from_template(
        """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Ты русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.
    Обязательно придерживайся следующих правил:
    1. Отвечай развернуто, используя данную тебе информацию.
    2. Ответ должен состоять из двух-трех предложений.
    3. Если ответа нет в данной информации, скажи что не знаешь ответа.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Тебе будет дана информация:
    {context} 
    Пожалуйста, ответь на этот вопрос : {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = load_llama()
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("Application startup successful!")

    # clear CLI
    os.system("cls" if os.name == "nt" else "clear")

    query = input("Добрый день! Чем я могу вам помочь?\n")
    while query:
        # print(len(sparse_retriever.invoke(query)))
        # print(len(dense_retriever.invoke(query)))
        # print(len(ensemble_retriever.invoke(query)))
        # print(retriever.invoke(query))
        rag_chain.invoke(query)
        print("\n", "-" * 100)
        query = input("Введите вопрос: ")
