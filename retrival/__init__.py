import logging
import time
LOGGER = logging.getLogger(__name__)

start = time.time()
LOGGER.debug(f"start init library")

from pathlib import Path
import json

from .core import get_embedding_core
from .qdrant_store import QdrantStore
# from .configs import cfg_db
from .prompt_builder import build_prompt as b_m

__all__ = [
    "Retreiver", "get_embedding_core"
]

LOGGER.debug(f"time init library: {time.time() - start} sec")
start = time.time()
LOGGER.debug(f"start init code")

class Retreiver():

    def __init__(self, embedding, collection_name: str = "migTorch"):
        self.db = QdrantStore(embedding, collection_name)

    def new_data(self, jsonl_patch: str|Path = "data/parsing/pytorch_chunks.jsonl"):
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
            self.db.add_texts(
                texts=texts[i : i + BATCH],
                metadatas=metadatas[i : i + BATCH],
            )
        LOGGER.debug('Данные успешно загружены в БД')
    
    def drop_and_create(self, new_collection_name, embedding = None):
        self.db.drop_and_create(new_collection_name, embedding)

    def _get_retriever_for_langChain(self, **kwargs):
        return self.db.vector_store.as_retriever(**kwargs)
    
    def build_prompt(self, old_code:str) -> str:
        context_docs = self._get_retriever_for_langChain(k=6).invoke(old_code)
        context_text = "\n\n---\n\n\n".join(doc.page_content 
                                            for doc in context_docs)
        prompt = b_m(old_code, context_text)
        return prompt
    

LOGGER.debug(f"time init code: {time.time() - start} sec")