import time
import logging

LOGGER = logging.getLogger(__name__)
start = time.time()
LOGGER.debug(f"start init library")

from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

LOGGER.debug(f"time init library: {time.time() - start} sec")

LOGGER.debug(f"start init code")
start = time.time()

class QdrantStore:
    """
        LangChain-vectorstore + Qdrant
    """
    def __init__(self,
            embedding,
            collection_name:str = "test_collection"
        ):
        self.embedding = embedding
        
        self.client = QdrantClient(url="http://localhost:6333")
        
        self._is_new_collection(collection_name)
        
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=embedding,
        )

    def _is_new_collection(self, collection_name: str):
        self.collection_name = collection_name
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name = self.collection_name,
                vectors_config=VectorParams(
                    size= self.embedding._client.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE)
            )

    def add_texts(self, texts: List[str], metadatas: List[Dict] | None = None) -> None:
        metadatas = metadatas or [{}] * len(texts)
        self.vector_store.add_texts(texts=list(texts), metadatas=list(metadatas))
        

    def similarity_search(self, query: str, k: int = 3):
        """
            Возвращаем k ближайших документов
        """
        return self.vector_store.similarity_search(query, k=k)

    def drop_and_create(self, new_name_collection: str,  
              embedding = None) -> None:
        """
            Полностью удалить коллекцию и создаём новую.
        """
        self.client.delete_collection(self.collection_name)
        LOGGER.debug(f'Коллекция {self.collection_name} удалена')
        if new_name_collection:
            self.collection_name = new_name_collection

        try:
            self.client.close()
        except Exception:
            pass
        
        if embedding:
            self.embedding = embedding

        self.client = QdrantClient(url="http://localhost:6333")

        self._is_new_collection(new_name_collection)

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding,
        )
        LOGGER.debug(f'Коллекция {self.collection_name} успешно создана')


LOGGER.debug(f"time init code: {time.time() - start} sec")