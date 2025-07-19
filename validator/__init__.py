import os 
import logging
import subprocess
import tempfile

LOGGER = logging.getLogger(__name__)

# Полный путь к conda.exe 
CONDA_PATH = 'C:/Miniconda3/Scripts/conda.exe'
DYNAMIC_VALIDATOR_PATH = os.path.abspath('validator/dynamic_validator.py')

# Проверка существования файла
if not os.path.exists(DYNAMIC_VALIDATOR_PATH):
    LOGGER.error(f"Файл dynamic_validator.py не найден по пути {DYNAMIC_VALIDATOR_PATH}. Создайте его в корне проекта.")
    exit
# Проверка существаования conda
if not os.path.exists(CONDA_PATH):
    LOGGER.error(f"Conda по пути {CONDA_PATH} не найдена.")
    exit

class Validator():

    def dynamic_val(self, code):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', 
                                         delete=False, # Будет ли автоматически удаляться файл 
                                         encoding='utf-8') as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name

        try:
            result_sub = subprocess.run(
                [CONDA_PATH, 'run', '-n', 'torch270', 'python', DYNAMIC_VALIDATOR_PATH, temp_path],
                capture_output=True, encoding='utf-8', check=False, timeout=120
            )
            print('Temp:', temp_path)
            dynamic_output = result_sub.stdout.strip()
            dynamic_errors = result_sub.stderr.strip()
            
            if result_sub.returncode != 0:
                dynamic_output += f"\nКритическая ошибка запуска ({temp_file}): {dynamic_errors}"
                LOGGER.error(dynamic_output)
            else:
                LOGGER.info(f"Динамическая валидация({temp_file}) прошла успешно.")
        except Exception as e:
            LOGGER.error(f"Неожиданная ошибка: {e}")
        # os.remove(temp_path)            

        return dynamic_output, dynamic_errors
