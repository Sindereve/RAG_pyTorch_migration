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

    def _dynamic_val(self, code, torch_version: str ='2.7.0'):
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
            LOGGER.error(f"Oшибка валидации: {e}")
        return out, err


    def _get_version_torch(self, conda_env_ending):
        try:
            version_cmd = [
                CONDA_PATH, 'run', '-n', f"torch{conda_env_ending}", 'python', '-c',
                'import torch; print(f"PyTorch version: {torch.__version__}")'
            ]
            out = subprocess.run(version_cmd, capture_output=True,
                                            encoding="utf-8").stdout.strip()
            return out
        except Exception as e: 
            LOGGER.error(f"_get_version_torch ERROR: {e}")
            raise

    def _version_in_conda_env_ending(self, version:str)->str:
        v_list = version.split('.')
        return ''.join(v_list)
    
    def run_test_code(self, code: str, env_version: str):
        '''
            Проверка работает ли код на версии env_version
        '''
        out, err = self._dynamic_val(code, torch_version=env_version)
        works = (err == '')
        return works, out, err
    

    def run_test_old_and_new_code(self, name_test, old_code, new_code):
        """
            Тестирование старого и нового кода
            :param name_test: Название теста
            :param old_code: Старвый код
            :param new_code: Новый код

        """
        LOGGER.info(f"[START TEST] {name_test}")
        
        works_100, out_100, err_100 = self.run_test_code(old_code, '1.0.0')
        LOGGER.info(f"{name_test} в 1.0.0: {'Работает' if works_100 else 'Не работает'}")
        LOGGER.debug(f"{name_test} в 1.0.0: {'Работает' if works_100 else 'Не работает'} | Out: {out_100} | Err: {err_100}")
        
        # Тест для 2.7.0
        works_270, out_270, err_270 = self.run_test_code(old_code, '2.7.0')
        LOGGER.info(f"{name_test} в 2.7.0: {'Работает' if works_270 else 'Не работает'}")
        LOGGER.debug(f"{name_test} в 2.7.0: {'Работает' if works_270 else 'Не работает'} | Out: {out_270} | Err: {err_270}")
        
        # Тест для ответа LLM
        new_code_works, new_code_out, new_code_err = self.run_test_code(new_code, '2.7.0')
        LOGGER.info(f"Вариант предложенный RAG в 2.7.0: {'Работает' if new_code_works else 'Не работает'}")
        LOGGER.debug(f"Вариант предложенный RAG в 2.7.0: {'Работает' if new_code_works else 'Не работает'} | Out: {new_code_out} | Err: {new_code_err}")
        
        LOGGER.info(f"[END TEST] {name_test}")

        return {
            'API': name_test,
            'Работает в 1.0.0': works_100,
            'Out 1.0.0': out_100,
            'Err 1.0.0': err_100,
            'Работает в 2.7.0': works_270,
            'Out 2.7.0': out_270,
            'Err 2.7.0': err_270[:200], # (в пандас есть ограничение по размеру) !!! возможно стоит сохранять ошибки другим способом!!!
            'Работает в 2.7.0 (после RAG)': new_code_works, 
            'RAG Out 2.7.0': new_code_out,
            'RAG Err 2.7.0': new_code_err,
        }, {
        'old_code': old_code,
            'new_code': new_code
        }
