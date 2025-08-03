
from langchain_core.prompts import PromptTemplate

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
First, provide the updated code in the following format:
```python
import torch
# updated code here
```
Then, after the code block, provide a detailed explanation as a single continuous paragraph of text describing the changes you made and the reasons for each change.

"""

PROMPT_TEMPLATE = PromptTemplate.from_template(_PROMPT_TXT.strip())

def build_prompt(old_code: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(old_code=old_code, context=context)

