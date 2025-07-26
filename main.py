from logs.logging_setup import configure_logging
configure_logging()

import logging
import time

LOGGER = logging.getLogger(__name__)
start = time.time()
LOGGER.debug(f"\nstart init library\n{'-'*30}")

from parsing import prepare
from retrival import Retreiver, get_embedding_core
from llm import get_llm_response, clean_llm_code
from validator import Validator
LLM_NAME = "tngtech/deepseek-r1t2-chimera"

LOGGER.debug(f"\ntime init library: {time.time() - start} sec\n{'-'*30}")

LOGGER.info("Старт проекта")

# =========  Загрузка модели =================
model, tokenizer = get_embedding_core("intfloat/e5-large")
LOGGER.info(f"Модель '{model.model_name}' загружена")

# =========  Создание ЧАНКОВ =================
# prepare(tokenizer, 512, 64)
# LOGGER.info("Чанки созданы")

# =========  Подключение к БД =================
rtr = Retreiver(model, model.model_name) 
LOGGER.info(f"БД подключена")


# =========  Загрузка данных в БД =================
# rtr.drop_all_data()
# LOGGER.info(f"Коллекция '{rtr.db.collection_name}' очищена от мусора")
# LOGGER.info("Начало загрузки данных")
# rtr.new_data()
# LOGGER.info(f"Данные успешно загружены в коллекцию '{rtr.db.collection_name}'")

# =========  Работа с retriv из long_chain =================
prompt = rtr.build_prompt("print(torch.tensor([[1., -1.], [1., -1.]]))")
LOGGER.info(f"Запрос в БД успешно выполнен")

result = get_llm_response(prompt, LLM_NAME)
LOGGER.info(f"Запрос в LLM успешно выполнен")

result_code = clean_llm_code(result)
LOGGER.info(f"Получен код из ответа")

val = Validator()
res, error = val.dynamic_val(result_code, torch_version="2.7.0")
LOGGER.info(f"Вывод от валидатора\nOut: {res}\nError: {error}")
LOGGER.info("Happy end")
