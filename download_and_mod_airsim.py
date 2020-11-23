import platform
from pathlib import Path
import requests
import zipfile
import os
from argparse import ArgumentParser
import stat

ENV_FOLDER = Path("gym_game_of_drones/envs/multi_agent/gym_airsimdroneracinglab")
DEST_AIRSIM_FOLDER = ENV_FOLDER / "ADRL"
# DEST_PAK_FOLDER = DEST_AIRSIM_FOLDER / "AirSim/AirSimExe/Content/Paks"

SYSTEM_STR = platform.system()
assert SYSTEM_STR == 'Linux' or SYSTEM_STR == 'Windows', f"ERROR: Found system: {SYSTEM_STR}. Compatible with Linux and Windows only."

if SYSTEM_STR == 'Linux':
    AIRSIM_DOWNLOAD_URL = "https://github.com/microsoft/AirSim-Drone-Racing-Lab/releases/download/v1.0-linux/ADRL.zip"

    # BUILDING_DOWNLOAD_URL = "https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/releases/download/v0.3.0-linux/Building99.pak"
    # SOCCER_DOWNLOAD_URL = "https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/releases/download/v0.3.0-linux/Soccer_Field.pak"
    # ZHANG_DOWNLOAD_URL = "https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/releases/download/v0.3.0-linux/ZhangJiaJie.pak"

else:  # Windows
    AIRSIM_DOWNLOAD_URL = "https://github.com/microsoft/AirSim-Drone-Racing-Lab/releases/download/v1.0-windows/ADRL.zip"

    # BUILDING_DOWNLOAD_URL = "https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/releases/download/v0.3.0/Building99.pak"
    # SOCCER_DOWNLOAD_URL = "https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/releases/download/v0.3.0/Soccer_Field.pak"
    # ZHANG_DOWNLOAD_URL = "https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/releases/download/v0.3.0/ZhangJiaJie.pak"


def download(src, dst):
    r = requests.get(src)
    with open(dst, 'wb') as f:
        f.write(r.content)


def mod_client_file_from_here():
    from gym_game_of_drones.modded_client.airsimdroneracinglab_mod import mod_client_file
    mod_client_file()


def run_install(args_main):
    dl = args_main.download
    if dl:
        print("Downloading the required AirSim binaries for your system (this can take a while):")
        print("Downloading ADRL.zip...")
        download(src=AIRSIM_DOWNLOAD_URL, dst=DEST_AIRSIM_FOLDER / "ADRL.zip")

        print("Unzipping ADRL.zip...")
        with zipfile.ZipFile(DEST_AIRSIM_FOLDER / "ADRL.zip", 'r') as zip_ref:
            zip_ref.extractall(path=DEST_AIRSIM_FOLDER)

        print("Deleting ADRL.zip...")
        os.remove(DEST_AIRSIM_FOLDER / "ADRL.zip")

        # print("Downloading Building99.pak...")
        # download(src=BUILDING_DOWNLOAD_URL, dst=DEST_PAK_FOLDER / "Building99.pak")
        #
        # print("Downloading Soccer_Field.pak...")
        # download(src=SOCCER_DOWNLOAD_URL, dst=DEST_PAK_FOLDER / "Soccer_Field.pak")
        #
        # print("Downloading ZhangJiaJie.pak...")
        # download(src=ZHANG_DOWNLOAD_URL, dst=DEST_PAK_FOLDER / "ZhangJiaJie.pak")

        for root, dir, filelist in os.walk('./'):
            for file in filelist:
                if "ADRL" in file and not ("pak" in file or "log" in file):
                    p = Path(root) / file
                    print(f"setting executable permission:{p}")
                    os.chmod(p, stat.S_IXUSR)

        print("AirSim is all set up!")

    print("Applying the modded client.py file to the airsimdroneracinglab library...")
    mod_client_file_from_here()

    print("All done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--download', dest='download', action='store_true')
    parser.add_argument('--no-download', dest='download', action='store_false')
    parser.set_defaults(download=True)
    args_main = parser.parse_args()
    run_install(args_main)
