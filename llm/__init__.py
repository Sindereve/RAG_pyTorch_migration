import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

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
