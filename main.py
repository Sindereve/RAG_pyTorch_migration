from logs.logging_setup import configure_logging
configure_logging()

import logging
import time
LOGGER = logging.getLogger(__name__)
start = time.time()
LOGGER.debug(f"\nstart init library\n{'-'*30}")

from parsing import prepare
from retrival import Retreiver, get_embedding_core
from llm import get_llm_response

LOGGER.debug(f"\ntime init library: {time.time() - start} sec\n{'-'*30}")

LOGGER.info("Старт проекта")

# =========  Создание ЧАНКОВ =================
# prepare(tokenizer, model_config.chunk_size, model_config.chunk_overlap)
# LOGGER.info("Чанки созданы")

# =========  Загрузка модели =================
model, tokenizer, model_config = get_embedding_core()
LOGGER.info(f"Модель {model_config.model_key} загружена")

# =========  Подключение к БД =================
rtr = Retreiver(model, "migTorch")
LOGGER.info(f"БД подключена")

# =========  Загрузка данных в БД =================
# rtr.new_data()
# LOGGER.info(f"Данные успешно загружены")

# =========  Работа с retriv из long_chain =================
prompt = rtr.build_prompt("torch.tensor([[1., -1.], [1., -1.]])")
LOGGER.info(f"Запрос в БД успешно выполнен")

result = get_llm_response(prompt)
LOGGER.info(f"Запрос в LLM успешно выполнен")
LOGGER.info(f"Результат:\n\n{result}")
