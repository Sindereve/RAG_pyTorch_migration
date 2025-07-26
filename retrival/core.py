import logging
import time
LOGGER = logging.getLogger(__name__)

start = time.time()
LOGGER.debug(f"start init library")

from functools import lru_cache     

LOGGER.debug(f"time init library: {time.time() - start} sec")

start = time.time()
LOGGER.debug(f"start init code")


@lru_cache
def get_embedding_core(name_model) -> tuple:
    """
    Создаёт один раз - embedder, tokenizer и splitter
    по параметрам из config.py  При повторных вызовах
    возвращает кешированную тройку (@lru_cache)
    
    :return модель, токенайзер:
    """
    
    from langchain_huggingface import HuggingFaceEmbeddings
    model = HuggingFaceEmbeddings(model=name_model)

    LOGGER.debug('Загружена embedder модель')
    # достаём токенайзер
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model.model_name,
                                            use_fast=True)
    LOGGER.debug('Токенайзер получен')
    return model, tok


LOGGER.debug(f"time init code: {time.time() - start} sec")