import logging
import time
LOGGER = logging.getLogger(__name__)

start = time.time()
LOGGER.debug(f"start init library")

from pathlib import Path
from .fetsh_github import get_releases, get_and_save_json as git_get_and_save_jsonl
from .chunker import chunk_jsonl as chunk_save_json

__all__ = [
    "get_releases",
    "git_get_and_save_jsonl", "prepare"
    "chunk_save_json",
]

LOGGER.debug(f"time init library: {time.time() - start} sec")
start = time.time()
LOGGER.debug(f"start init code")

def prepare(
    tokenizer,
    chunk_size: int, overlap: int,
    out_dir: str | Path = "data/parsing",
    min_version: str = "v1.0.0", max_version: str = "v2.7.0"
    ) -> None:
    """
    Процесс скачивания всех данных и разделения на чанки
    * Скачиваем release notes и сохраняем в ``<out_dir>/changelog_pytorch.jsonl``
    * Разделяем на чанки и сохраняем в ``<out_dir>/pytorch_chunks.jsonl``
    """
    out_dir = Path(out_dir)
    raw_path = out_dir / "changelog_pytorch.jsonl"
    chunks_path = out_dir / "pytorch_chunks.jsonl"


    git_get_and_save_jsonl(
        min_version=min_version, 
        max_version=max_version, 
        filename=raw_path,
    )
    chunk_save_json(
        raw_path,
        chunks_path,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    
    LOGGER.debug("Данные успешно скачены. Чанки созданы.")
    return None

LOGGER.debug(f"time init code: {time.time() - start} sec")