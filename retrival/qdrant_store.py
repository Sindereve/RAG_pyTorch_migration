import time
import logging

LOGGER = logging.getLogger(__name__)
start = time.time()
LOGGER.debug(f"start init library")

from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FilterSelector
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings

LOGGER.debug(f"time init library: {time.time() - start} sec")

LOGGER.debug(f"start init code")
start = time.time()

class QdrantDB:
    """
        Класс для взаимодействия с Qdrant.
    """
    
    def __init__(
        self,
        embedding: Embeddings,
        collection_name: str = "base",
        qdrant_url: str = "http://localhost:6333",
        ):
        """
            Клиент Qdrant и векторного стора.

            :param embedding: модель эмбеддингов.
            :param collection_name: имя коллекции в Qdrant.
            :param qdrant_url: Qdrant.
        """
        self.embedding = embedding
        self.collection_name = collection_name
        
        self.client = QdrantClient(
            url=qdrant_url,
        )
        
        self._is_collection_exists()
        
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding,
        )
    
    def _is_collection_exists(self) -> None:

        LOGGER.debug(f"Проверяем коллекцию: {self.collection_name}")
        if not self.client.collection_exists(self.collection_name):
            vector_size = len(self.embedding.embed_query("test"))
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE  # можно будет потом поэксперементировать с методом поиска в БД
                )
            )
            LOGGER.info(f"Создана новая коллекция: {self.collection_name}")
        else:
            LOGGER.debug(f"Коллекция {self.collection_name} уже существует")
    
    def add_texts(self, texts: List[str], metadatas: List[Dict] = None) -> None:
        """
            Добавляет тексты в БД с генерацией эмбеддингов и метаданными.
        
            :param texts: список текстов для добавления
            :param metadatas: список словарей с метаданными
        """
        metadatas = metadatas or [{}] * len(texts) 
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        
    
    def drop_all_data(self) -> None:
        """
            Удаляет все данные из коллекции, но сохраняет саму коллекцию.
        """
        LOGGER.debug(f"Удаляем все данные из коллекции {self.collection_name}")
        
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter()  # когда, забыл where при delete
            ),
            wait=True
        )
        
        LOGGER.debug("Удаление завершено")

    def drop_all_collections(self) -> None:
        LOGGER.warning("Удаляем все коллекции в Qdrant")
        
        collections = self.client.get_collections().collections
        for collection in collections:
            collection_name = collection.name
            LOGGER.debug(f"Удаляем коллекцию: {collection_name}")
            self.client.delete_collection(collection_name)
        
        LOGGER.warning("Все коллекции удалены")

        self.collection_name = "base"
        self._is_collection_exists()
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding,
        )
        warning_sms = f"Все коллекции были удалены. Создана новая базовая коллекция 'base'. \
                       \n - embed name: {self.embedding.model_name} \
                       \n- collection name: {self.collection_name}"
        LOGGER.warning(warning_sms)  
    
    def similarity_search(self, query: str, 
                          k: int = 5, 
                          score_threshold: float = None
        ) -> List[Dict]:
        """
            Поиск по запросу.
        
            :param query: запрос
            :param k: количество результатов.
            :param score_threshold: порог похожести.
            :return: список словарей с документами, метаданными и скором.
        """
        LOGGER.debug(f"Поиск по запросу '{query}' в коллекции {self.collection_name}")
        
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            score_threshold=score_threshold
        )
        
        formatted_results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            } for doc, score in results
        ]
        
        LOGGER.debug(f"Найдено {len(formatted_results)} результатов")
        return formatted_results
    
    def get_collection_info(self) -> Dict:
        """
            Возвращает информацию о коллекции
        """
        info = self.client.get_collection(self.collection_name)
        return {
            "points_count": info.points_count,
            "status": info.status,
            "vectors_count": info.vectors_count,
        }


LOGGER.debug(f"time init code: {time.time() - start} sec")



if __name__ == "__main__":
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    db = QdrantDB(embedding=embedding_model, collection_name="test_rag")
    
    # Добавляем данные
    texts = ["Пример текста 1", "Пример текста 2"]
    metadatas = [{"source": "doc1"}, {"source": "doc2"}]
    db.add_texts(texts, metadatas)
    
    # Поиск
    results = db.similarity_search("Пример", k=2)
    print(results)
    
    # Удаление всех данных в текущей коллекции
    db.drop_all_data()
    
    # Удаление всех коллекций
    db.drop_all_collections()