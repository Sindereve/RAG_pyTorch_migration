# RAG_pyTorch_migration

Проект Retrieval-Augmented Generation [(RAG)](https://habr.com/ru/articles/779526/) для миграции кода с версии v1.0.0 PyTorch на v2.7.0. 

## Особенности 

* langchain в связки с HuggingFaceEmbeddings
* векторная БД QDrant
* работа с API LLM через [openrouter](https://openrouter.ai/)

## Структура проекта
``` text
RAG_pyTorch_migration/
├── data/                  # Примеры документов и наборов данных
├── documents/             # Рекомендации по установке и настройке
├── llm/                   # Функции для работы с LLM
├── logs/                  # Логи и настройка логера
├── parsing/               # Модуль для скачивания данных и разбиения на чанки
├── retrival/              # модуль кетривал который отвечает за взаимодействие с DB
|                                # и загрузку embed модели
├── validator/             # Валидатор для проверки работоспособности кода
├── .gitignore        
├── main.py                # Пример полного пайплайна 
├── main_jup.py            # Блокнот с представлением проекта
├── testing_differen_params.py  # Блок для тестирования
└── README.md  
```
