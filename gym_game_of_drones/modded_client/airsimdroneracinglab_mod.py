import os
import pathlib
import shutil
import airsimdroneracinglab


def mod_client_file():
    as_path = pathlib.Path(airsimdroneracinglab.__file__).resolve().parent
    dir_path = os.path.dirname(os.path.realpath(__file__))
    mod_path = os.path.join(dir_path, 'client.py')
    target_path = os.path.join(as_path, 'client.py')
    shutil.copyfile(src=mod_path, dst=target_path)
