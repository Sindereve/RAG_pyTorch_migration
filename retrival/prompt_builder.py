
from langchain_core.prompts import PromptTemplate

# prompt_tmpl = PromptTemplate.from_template(
_PROMPT_TXT="""
You are a senior PyTorch engineer helping migrate code from v1.0 to v2.7.

### Original code (v1.0)
```python
{old_code}
```

### Context (docs & changelog)
{context}

### Task
Rewrite the code so it runs on **PyTorch 2.7**. Explain what you changed.

### Answer format
```python
import torch
# updated code here
```
"""

PROMPT_TEMPLATE = PromptTemplate.from_template(_PROMPT_TXT.strip())

def build_prompt(old_code: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(old_code=old_code, context=context)

