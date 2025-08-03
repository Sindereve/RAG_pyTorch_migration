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
import json
import pandas as pd

LOGGER.debug(f"\ntime init library: {time.time() - start} sec\n{'-'*30}")

LOGGER.info("START PROJ")

model_names = ["sentence-transformers/all-MiniLM-L6-v2", 
               "BAAI/bge-base-en", 
               "BAAI/bge-large-en", 
               "intfloat/e5-large", 
               "sentence-transformers/LaBSE"
]

LLM_NAME = "tngtech/deepseek-r1t2-chimera"
PATH_TEST_DICT = 'validator/old_code.json'

val = Validator()

for model_name in model_names:
    # =========  Загрузка модели =================
    LOGGER.info(f"\n{'_'*50}\n-- ПРОВЕРКА МОДЕЛИ {model_name}\n{'_'*50}")
    model, tokenizer = get_embedding_core(model_name)
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
    # =========  На данный момент все нужные данные у нас уже есть =================


    with open(PATH_TEST_DICT, 'r') as file:
        testing_data = json.load(file)


    info_list = []
    number_test = 0
    for name, old_code in testing_data.items():
        number_test+=1
        LOGGER.info(f"[Start] Test {number_test}")
        LOGGER.info(f"- Запрос в БД")
        prompt = rtr.build_prompt(old_code)

        LOGGER.info(f"- Запрос в LLM")
        result = get_llm_response(prompt, LLM_NAME)

        LOGGER.info(f"- Очистка ответа LLM")
        new_code, explanation = clean_llm_code(result)

        LOGGER.info(f"- Валидация")
        info_dict, _ = val.run_test_old_and_new_code(name, old_code, new_code, explanation)
        info_list.append(info_dict)
        LOGGER.info(f"[End] Test {number_test}")
        
    dfTest = pd.DataFrame(info_list)
    dfTest.to_csv(f"data/{model.model_name.replace('/', '_')}.csv")
    LOGGER.info(f"Файл с результатами сохранён в data/{model.model_name.replace('/', '_')}.csv")


LOGGER.info("HAPPY END")