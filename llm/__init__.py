import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LOGGER = logging.getLogger(__name__)

CLIENT = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API'),
)

def get_llm_response(prompt: str, model_name:str = "moonshotai/kimi-k2:free")-> str:
    completion = CLIENT.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

import re  

def clean_llm_code(raw_output):
    """
        Очистка вывода LLM: извлекает Python-код из Markdown, удаляет артефакты.
    """
    
    code_match = re.search(r'```(?:python)?\s*\n?(.*?)```', raw_output, re.DOTALL | re.IGNORECASE)
    if code_match:
        code = code_match.group(1).strip()
    else:
        code = raw_output.strip()
        LOGGER.warning("Не найден Markdown-блок с кодом в LLM-ответе")
    
    code = code.replace('\u2192', '->')
    code = code.replace('\u2014', '-')
    
    code = '\n'.join(line.rstrip() for line in code.splitlines() if line.strip())
    LOGGER.debug('Текст из LLM очищен. Оставлен только код')
    return code