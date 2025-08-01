import logging
import time
LOGGER = logging.getLogger(__name__)

start = time.time()
LOGGER.debug(f"start init library")

from pathlib import Path
import json

from .core import get_embedding_core
from .qdrant_store import QdrantDB
from .prompt_builder import build_prompt as b_m
from langchain_core.embeddings import Embeddings

__all__ = [
    "Retreiver", "get_embedding_core"
]

LOGGER.debug(f"time init library: {time.time() - start} sec")
start = time.time()
LOGGER.debug(f"start init code")

class Retreiver():
    """
        Класс для работы с ретривером
    """

    def __init__(self, embedding: Embeddings, collection_name: str = "base"):
        if collection_name != "base":
            collection_name = f"migTorch_test_{collection_name.replace('/', '_')}"
        self.db = QdrantDB(embedding, collection_name)

    def new_data(self, jsonl_patch: str|Path = "data/parsing/pytorch_chunks.jsonl") -> None:
        jsonl_path = Path(jsonl_patch)
        texts, metadatas = [], []
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj["content"])         
                metadatas.append({
                    "version": obj["version"],
                    "release_name": obj["release_name"],
                    "chunk_id": obj["chunk_id"],
                })
        BATCH = 200
        for i in range(0, len(texts), BATCH):
            LOGGER.debug(f"Добавляем текста[{i}:{i + BATCH}] в коллекцию {self.db.collection_name}")
            self.db.add_texts(
                texts=texts[i : i + BATCH],
                metadatas=metadatas[i : i + BATCH],
            )
        LOGGER.debug('Данные успешно загружены в БД')

    def _get_retriever_for_langChain(self, **kwargs):
        return self.db.vector_store.as_retriever(**kwargs)
    
    def build_prompt(self, old_code:str) -> str:
        context_docs = self._get_retriever_for_langChain(k=6).invoke(old_code)
        LOGGER.debug('Получен ответ от БД')
        context_text = "\n\n---\n\n\n".join(doc.page_content 
                                            for doc in context_docs)
        prompt = b_m(old_code, context_text)
        LOGGER.debug('Промт создан')
        return prompt

    # Методы для очистки
    def drop_all_data(self) -> None:
        self.db.drop_all_data()

    def drop_all_collections(self) -> None:
        self.db.drop_all_collections()

    def switch_to(self, new_embedding: Embeddings, new_collection_name: str) -> None:
        """
            Переключение на другую коллекцию с другой моделью 
        """
        LOGGER.debug(f"Переключаемся на новую коллекцию '{new_collection_name}' с моделью {new_embedding.model_name}")
        self.db = QdrantDB(embedding=new_embedding, collection_name=new_collection_name)
        LOGGER.debug("Переключение завершено")


LOGGER.debug(f"time init code: {time.time() - start} sec")