import logging
import time
LOGGER = logging.getLogger(__name__)

start = time.time()
LOGGER.debug(f"start init library")

from functools import lru_cache
from typing import List, Protocol
from .configs import cfg_emb            

LOGGER.debug(f"time init library: {time.time() - start} sec")

start = time.time()
LOGGER.debug(f"start init code")

# Father
class Embedder(Protocol):

    dim:int  # размерность вектора

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
            Вернуть список эмбеддингов (list[list[float]]).
        """

#  счастливое лицо :D
class HFEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 batch_size=32):
        self.model_name = model_name
        self.batch_size = batch_size

        from langchain_huggingface import HuggingFaceEmbeddings
        self.model = HuggingFaceEmbeddings(model=model_name)
        
        LOGGER.debug('Модель создана.')

    def __repr__(self):
        return f"HFEmbedder(model_name={self.model_name!r}, batch_size={self.batch_size})"


LIBRARY_MODEL = {
    "mini":      lambda: HFEmbedder("sentence-transformers/all-MiniLM-L6-v2"),
    "bge-base":  lambda: HFEmbedder("BAAI/bge-base-en"),
    "bge-large": lambda: HFEmbedder("BAAI/bge-large-en"),
    "e5-large":  lambda: HFEmbedder("intfloat/e5-large"),
    "labse":     lambda: HFEmbedder("sentence-transformers/LaBSE"), 
}

@lru_cache
def get_embedding_core() -> tuple:
    """
    Создаёт один раз - embedder, tokenizer и splitter
    по параметрам из config.py  При повторных вызовах
    возвращает кешированную тройку (@lru_cache)
    
    :return модель, токенайзер, конфиг:
    """
    hfemb = LIBRARY_MODEL[cfg_emb.model_key]()
    
    LOGGER.debug('Загружена embedder модель')
    # достаём токенайзер
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hfemb.model_name,
                                            use_fast=True)
    LOGGER.debug('Токенайзер получен')
    return hfemb.model, tok, cfg_emb


LOGGER.debug(f"time init code: {time.time() - start} sec")