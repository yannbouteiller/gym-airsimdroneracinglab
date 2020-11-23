from init_ip_port_and_json_files import init_ip_port_and_json_files
from argparse import ArgumentParser
import os
import subprocess

EXECUTABLE_NAME = 'AirSimExe.sh'
OPTIONS = '-windowed -BENCHMARK -opengl4'


def main(args):
    clockspeed = args.clockspeed
    img_width = args.img_width
    img_height = args.img_height
    viewmode = args.viewmode
    init_ip_port_and_json_files(clockspeed, img_width=img_width, img_height=img_height, viewmode=viewmode)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, 'environments', 'gym-airsimdroneracinglab', 'gym_airsimdroneracinglab', 'envs', 'AirSim', EXECUTABLE_NAME)
    subprocess.Popen([dir_path, OPTIONS])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--clockspeed', type=float, default=1.0)
    parser.add_argument('--img_width', type=int, default=320)
    parser.add_argument('--img_height', type=int, default=240)
    parser.add_argument('--viewmode', type=str, default="FlyWithMe")
    args = parser.parse_args()
    main(args)
