import logging
import time
LOGGER = logging.getLogger(__name__)
start = time.time()
LOGGER.debug(f"start init library")

import json
from pathlib import Path
from typing import List
# !!!! langchain.text_splitter ПОМОТРЕСТЬ !!!!

LOGGER.debug(f"time init library: {time.time() - start} sec")
start = time.time()
LOGGER.debug(f"start init code")

def _split_text(text: str, tokenizer, 
                 chunk_size: int, overlap: int, log: bool = False) -> List[str]:
    safe_len = tokenizer.model_max_length - 2

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=min(chunk_size, safe_len),
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    from transformers import logging as hf_log
    hf_log.set_verbosity_error()
    chunks = splitter.split_text(text)
    hf_log.set_verbosity_warning()

    for chunk in chunks:
        n = len(tokenizer(chunk)['input_ids'])
        if n > tokenizer.model_max_length:
            LOGGER.warn(f" {n} > {tokenizer.model_max_length} {chunk[:20]}")
    return chunks 

def chunk_jsonl(input_path: str, output_path: str, 
    tokenizer, 
    chunk_size: int = 512, overlap: int = 64) -> None:

    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved = 0

    with in_path.open("r", encoding="utf-8", errors="ignore") as infile, out_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line_num, line in enumerate(infile):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning(f"Ошибка в строчке {line_num} jsonl")
                continue

            tag = item.get("tag")
            name = item.get("name")
            body = item.get("body") or item.get("text")
            url = item.get("url")

            if not body:
                continue

            for idx, chunk in enumerate(
                _split_text(body, tokenizer, chunk_size=chunk_size, overlap=overlap)
            ):
                chunk_entry = {
                    "version": tag,
                    "release_name": name,
                    "chunk_id": f"{tag}-{idx}",
                    "content": chunk,
                    "source_url": url,
                }
                json.dump(chunk_entry, outfile, ensure_ascii=False)
                outfile.write("\n")
                
                saved += 1
    LOGGER.debug(f"Сохранено {saved} чанков -> {out_path}")
    return None

LOGGER.debug(f"time init code: {time.time() - start} sec")