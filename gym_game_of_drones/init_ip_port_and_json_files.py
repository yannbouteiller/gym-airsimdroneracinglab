import pickle
from gym_game_of_drones.envs.multi_agent.gym_airsimdroneracinglab.custom_airsim_settings_creator import CustomAirSimSettingsCreator
from gym_game_of_drones.envs.multi_agent.gym_airsimdroneracinglab.airsimdroneracinglab_env import initialize_ip_port_file, DEFAULT_IP_PORT_FILE_NAME
from argparse import ArgumentParser


def init_ip_port_and_json_files(clockspeed=1.0, img_width=320, img_height=240, viewmode="FlyWithMe"):
    initialize_ip_port_file()
    f = open(DEFAULT_IP_PORT_FILE_NAME, 'rb')
    ip_port = pickle.load(f)
    f.close()
    port = ip_port.global_port_count
    ip_str = f"{ip_port.global_ip_count_4}.{ip_port.global_ip_count_3}.{ip_port.global_ip_count_2}.{ip_port.global_ip_count_1}"
    CustomAirSimSettingsCreator().write_custom_settings_file(clockspeed=clockspeed, ip=ip_str, port=port, img_width=img_width, img_height=img_height, viewmode=viewmode)


def main(args):
    clockspeed = args.clockspeed
    init_ip_port_and_json_files(clockspeed=clockspeed)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--clockspeed', type=float, default=1.0)
    args = parser.parse_args()
    main(args)
