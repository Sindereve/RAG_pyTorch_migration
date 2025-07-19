import sys
import subprocess
import logging
import torch

LOGGER = logging.getLogger(__name__)

# Кодировка для win
if sys.stdout.encoding != 'UTF-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def main(code_file):
    try:
        exec_result = subprocess.run(['python', code_file], capture_output=True, encoding='utf-8', timeout=90)
        if exec_result.returncode == 0:
            print(f"Output: {exec_result.stdout.strip()}")
        else:
            print(exec_result.stderr.strip())
    except Exception as e:
        print(f"Неожиданная ошибка в валидации: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    main(sys.argv[1])