import requests
import json
import logging

API_URL = "https://api.github.com/repos/pytorch/pytorch/releases"
HEADERS = {
    "Accept": "application/vnd.github+json"
}

LOGGER = logging.getLogger(__name__)

def get_releases() -> list:
    '''
        Получаем все данные об изменениях в версиях
        https://github.com/pytorch/pytorch/releases
    '''
    releases = []
    page = 1
    while True:
        response = requests.get(f"{API_URL}?page={page}&per_page=100", headers=HEADERS)
        data = response.json()
        if not data:
            break
        releases.extend(data)
        page += 1
    return releases

def _parse_version(version_str: str) -> tuple:
    version_list = version_str.lstrip("v").split('.')
    return tuple(int(p) for p in version_list) 

def _is_version(tag_str: str, tag_min: str | None = None, tag_max: str | None = None) -> bool:
    """
        Одна из нужных версий
    """
    try:
        tag_turp = _parse_version(tag_str)
        tag_min = _parse_version(tag_min) if tag_min else (0, 0, 0)
        tag_max = _parse_version(tag_max) if tag_max else (9, 9, 9)
        return tag_min < tag_turp <= tag_max
    except ValueError:
        return False

def save_as_jsonl(releases: list, 
                  min_version: str, 
                  max_version: str, 
                  out_file_path: str = "data/parsing/changelog_pytorch.jsonl"):
    """
        Сохраняем информацию об изменениях в .jsonl
        
        :return: Количество сохранённых версий
    """
    
    release_save_count = 0
    with open(out_file_path, "w", encoding="utf-8") as f:
        for release in releases:
            if not _is_version(release.get("tag_name"), min_version, max_version):
                continue
            item = {
                "tag": release.get("tag_name"),
                "name": release.get("name"),
                "body": release.get("body"),
                "url": release.get("html_url")
            }
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
            release_save_count+=1
    return release_save_count, out_file_path

def get_and_save_json(min_version: str = "v1.0.0", 
                      max_version: str = "v2.7.0", 
                      filename: str = "changelog_pytorch.jsonl"):
    
    LOGGER.debug("Скачиваем PyTorch releases…")
    releases = get_releases()
    LOGGER.debug(f"Скачено {len(releases)} релизов…")
    release_save_count, file_patch = save_as_jsonl(releases, min_version, max_version, filename)
    LOGGER.debug(f"Сохранено {release_save_count} релизов в файле {file_patch}")

if __name__ == "__main__":
    get_and_save_json()