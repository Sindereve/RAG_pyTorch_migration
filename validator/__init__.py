import logging
import time
LOGGER = logging.getLogger(__name__)
start = time.time()
LOGGER.debug(f"start init library")

import os 
import logging
import subprocess
import tempfile

# Полный путь к conda.exe 
CONDA_PATH = 'C:/Miniconda3/Scripts/conda.exe'

# Проверка существаования conda
if not os.path.exists(CONDA_PATH):
    LOGGER.error(f"Conda по пути {CONDA_PATH} не найдена.")
    exit

LOGGER.debug(f"time init library: {time.time() - start} sec")


class Validator():

    def dynamic_val(self, code, torch_version: str ='2.7.0'):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', 
                                         delete=False, # Будет ли автоматически удаляться файл 
                                         encoding='utf-8') as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name

        try:
            LOGGER.debug(f"Начало валидации")
            conda_env_ending = self._version_in_conda_env_ending(torch_version)
            LOGGER.debug(f"In env: {self._get_version_torch(conda_env_ending)}")

            runner_config = [CONDA_PATH, 'run', '-n', f'torch{conda_env_ending}', 'python', temp_path]
            result_sub = subprocess.run(runner_config, capture_output=True,
                                        encoding='utf-8')

            out = result_sub.stdout.strip()
            err = result_sub.stderr.strip()

            if result_sub.returncode != 0:
                LOGGER.error(f"Oшибка при исполнении code: {err}")
            else:
                LOGGER.debug(f'Валидация успешно проведена файл({temp_path})')
        except Exception as e:
            LOGGER.error(f"Oшибка: {e}")
        return out, err


    def _get_version_torch(self, conda_env_ending):
        version_cmd = [
            CONDA_PATH, 'run', '-n', f"torch{conda_env_ending}", 'python', '-c',
            'import sys, torch; sys.stdout.reconfigure(encoding="utf-8"); print(f"PyTorch version: {torch.__version__}")'
        ]
        return subprocess.run(version_cmd, capture_output=True,
                                        encoding="utf-8", timeout=10).stdout.strip()

    def _version_in_conda_env_ending(self, version:str)->str:
        v_list = version.split('.')
        return ''.join(v_list)
