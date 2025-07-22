import os
import logging
import re # для очистки кода
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, OpenAIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

LOGGER = logging.getLogger(__name__)

CLIENT = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API'),
    timeout=30,
)


@retry(
    retry=retry_if_exception_type(APIConnectionError), # Retry только для connection ошибок
    stop=stop_after_attempt(5),  # 5 попыток
    wait=wait_exponential(multiplier=1, min=2, max=30),  # ждать 2, 4, 8, 16, 30 сeк
    reraise=True  # Перебросить оригинальную ошибку после retries
)
def get_llm_response(prompt: str, model_name: str = "deepseek/deepseek-chat:free")-> str:
    try:
        completion = CLIENT.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content  
    except OpenAIError as e:
        if getattr(e, 'code') == 429:
            raise OpenAIError('Rate limit!!!')
        raise
    

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