import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from logs.logging_setup import configure_logging
import logging
import time
from retrival import Retreiver, get_embedding_core
from llm import get_llm_response, clean_llm_code, translate
from validator import Validator

# Настройка логирования
configure_logging()
LOGGER = logging.getLogger(__name__)

model_names = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en",
    "BAAI/bge-large-en",
    "intfloat/e5-large",
    "sentence-transformers/LaBSE"
]
languages = ["English", "German", "Russian", "Spanish", "Japanese", "Portuguese"]

st.title("Миграция v1.0.0 -> v2.7.0 PyTorch")
st.write("**Настройка**")

# Выбор модели эмбеддингов
selected_embed_model = st.selectbox(
    "Выберите модель эмбеддингов:",
    options=model_names,
    index=model_names.index("intfloat/e5-large")
)

# Ввод LLM_NAME
default_llm_name = "qwen/qwen3-coder"
llm_name = st.text_input(
    "Введите название LLM модели:",
    value=default_llm_name
)

# Выбор языка для перевода
translation_language = st.selectbox(
    "Выберите язык для перевода описания:",
    options=languages,
    index=languages.index("English")
)

# Инициализация модели
@st.cache_resource
def load_model(selected_embed_model):
    start = time.time()
    LOGGER.debug(f"\nstart init library\n{'-'*30}")
    model, tokenizer = get_embedding_core(selected_embed_model)
    LOGGER.debug(f"\ntime init library: {time.time() - start} sec\n{'-'*30}")
    LOGGER.info(f"Модель '{model.model_name}' загружена")
    rtr = Retreiver(model, model.model_name)
    LOGGER.info(f"БД подключена")
    return model, tokenizer, rtr

model, tokenizer, rtr = load_model(selected_embed_model)

# Текстовое поле для ввода кода
code_input = st.text_area("Введите код для обработки и валидации:", height=200, placeholder="Например: print(torch.tensor([[1., -1.], [1., -1.]]))")

# Переменные для хранения результатов обработки
if "result_code" not in st.session_state:
    st.session_state.result_code = None
if "explanation" not in st.session_state:
    st.session_state.explanation = None
if "res" not in st.session_state:
    st.session_state.res = None
if "error" not in st.session_state:
    st.session_state.error = None
if "translated_explanation" not in st.session_state:
    st.session_state.translated_explanation = None

# Кнопка для запуска обработки
if st.button("Обработать код"):
    if code_input.strip():
        with st.spinner("Обработка кода..."):
            
            # Построение промпта
            prompt = rtr.build_prompt(code_input)
            LOGGER.info(f"Запрос в БД успешно выполнен")

            # Получение ответа от LLM
            result = get_llm_response(prompt, llm_name)
            LOGGER.info(f"Запрос в LLM успешно выполнен")

            # Очистка кода и объяснения
            result_code, explanation = clean_llm_code(result)
            LOGGER.info(f"Получен код и объяснение из ответа")

            # Валидация кода
            val = Validator()
            res, error = val.dynamic_val(result_code, torch_version="2.7.0")
            LOGGER.info(f"Вывод от валидатора\nOut: {res}\nError: {error}")

            # Сохранение результатов в session_state
            st.session_state.result_code = result_code
            st.session_state.explanation = explanation
            st.session_state.res = res
            st.session_state.error = error
            st.session_state.translated_explanation = None  # Сброс перевода
    else:
        st.warning("Пожалуйста, введите код для обработки")

# Отображение результатов, если они есть
if st.session_state.result_code or st.session_state.explanation or st.session_state.res or st.session_state.error:
    st.subheader("Результаты работы RAG")
    st.write("**Код:**")
    st.code(st.session_state.result_code if st.session_state.result_code else "LLM не удалось обработать код")
    st.write("**Вывод:**")
    st.code(st.session_state.res if st.session_state.res else "Нет вывода")
    st.write("**Ошибки:**")
    st.code(st.session_state.error if st.session_state.error else "Ошибок нет")
    st.write("**Описание:**")
    st.write(st.session_state.explanation if st.session_state.explanation else "Описания нет")

# Кнопка для перевода
if st.button("Перевести описание"):
    if st.session_state.explanation:
        with st.spinner("Перевод описания..."):
            try:
                st.session_state.translated_explanation = translate(
                    st.session_state.explanation, translation_language, llm_name
                )
                LOGGER.info(f"Описание переведено на {translation_language}")
            except Exception as e:
                st.error(f"Ошибка при переводе: {str(e)}")
                LOGGER.error(f"Ошибка при переводе описания: {str(e)}")
    else:
        st.warning("Нет описания для перевода")

# Отображение переведенного описания, если оно есть
if st.session_state.translated_explanation:
    st.subheader(f"Переведенное описание ({translation_language})")
    st.write(st.session_state.translated_explanation if st.session_state.translated_explanation else "Перевод не выполнен")