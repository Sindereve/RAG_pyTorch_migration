import logging
import time
LOGGER = logging.getLogger(__name__)
start = time.time()
LOGGER.debug(f"start init library")

import os 
import sys
import logging
import subprocess
import tempfile

# Patch for conda.exe 
CONDA_PATH = 'C:/Miniconda3/Scripts/conda.exe'
if not os.path.exists(CONDA_PATH):
    LOGGER.error(f"Conda {CONDA_PATH} .")
    sys.exit()

LOGGER.debug(f"time init library: {time.time() - start} sec")

class Validator():

    def _dynamic_val(self, code, torch_version: str ='2.7.0', is_delet_temp:bool = False):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', 
                                         delete=is_delet_temp, # NO AUTO DELETE !!
                                         encoding='utf-8') as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name

        try:
            LOGGER.debug(f"Start validation")
            conda_env_ending = self._version_in_conda_env_ending(torch_version)
            LOGGER.debug(f"In env: {self._get_version_torch(conda_env_ending)}")

            runner_config = [CONDA_PATH, 'run', '-n', f'torch{conda_env_ending}', 'python', temp_path]
            result_sub = subprocess.run(runner_config, capture_output=True,
                                        encoding='utf-8')

            out = result_sub.stdout.strip()
            err = result_sub.stderr.strip()

            if result_sub.returncode != 0:
                LOGGER.error(f"ERROR in conda: {err}")
            else:
                LOGGER.debug(f'ERROR in code: {temp_path})')
            return out, err
        except Exception as e:
            LOGGER.error(f"ERROR in validation: {e}")
        

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
    
    def run_test_code(self, code: str, env_version: str, 
                      is_delet_temp:bool = False):
        """
            Check works code on version env_version
        """
        out, err = self._dynamic_val(code, torch_version=env_version, is_delet_temp=is_delet_temp)
        works = (err == '')
        return works, out, err
    
    def run_test_old_and_new_code(self, name_test, old_code, new_code, is_delet_temp: bool = False):
        """
            Testing old and new code
        """
        LOGGER.info(f"[START TEST] {name_test}")
        
        works_100, out_100, err_100 = self.run_test_code(old_code, '1.0.0', is_delet_temp)
        info = f"{name_test} in 1.0.0: {'Work' if works_100 else 'Not work'}"
        LOGGER.info(info)
        LOGGER.debug(f"{info} | Out: {out_100} | Err: {err_100}")
        
        # Test 2.7.0
        works_270, out_270, err_270 = self.run_test_code(old_code, '2.7.0', is_delet_temp)
        info = f"{name_test} in 2.7.0: {'Work' if works_270 else 'Not work'}"
        LOGGER.info(info)
        LOGGER.debug(f"{info} | Out: {out_270} | Err: {err_270}")
        
        # Test answer LLM
        new_code_works, new_code_out, new_code_err = self.run_test_code(new_code, '2.7.0', is_delet_temp)
        info = f"Answer LLM RAG in 2.7.0: {'Work' if new_code_works else 'Not work'}"
        LOGGER.info(info)
        LOGGER.debug(f"{info} | Out: {new_code_out} | Err: {new_code_err}")
        
        LOGGER.info(f"[END TEST] {name_test}")

        return {
            'API': name_test,
            'Is work in 1.0.0': works_100,
            'Out 1.0.0': out_100,
            'Err 1.0.0': err_100[:200], # Think need record errors some other way. In future will change.
            'Is work in 2.7.0': works_270,
            'Out 2.7.0': out_270,
            'Err 2.7.0': err_270[:200], # Think need record errors some other way. In future will change.
            'Is work in 2.7.0 (после RAG)': new_code_works, 
            'RAG Out 2.7.0': new_code_out,
            'RAG Err 2.7.0': new_code_err[:200], # Think need record errors some other way. In future will change.
        }, {
        'old_code': old_code,
            'new_code': new_code
        }
