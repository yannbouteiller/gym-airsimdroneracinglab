from argparse import ArgumentParser
import airsimdroneracinglab as airsim
# import keyboard
import numpy as np
import gym
# import cv2
import pygame as pg
from pathlib import Path
from pyinstrument import Profiler
from platform import system
from collections import deque
import pickle
import os
import copy
import time

np.set_printoptions(precision=2, suppress=True, sign=' ', floatmode='fixed')

SYS_STR = system()
print(f"INFO: detected system: {SYS_STR}")

# pygame constants:

PG_DISPLAY_WIDTH = 800
PG_DISPLAY_HEIGHT = 600
PG_GAME_FOLDER_PATH = Path("drone_racing_dataset_collector")
PG_BACKGROUND_COLOR = (0, 0, 0)
PG_IMG_BACKGROUND_COLOR = (255, 255, 255)
PG_TEXT_COLOR = (255, 255, 255)
PG_GAME_CAPTION = 'Drone Racing Dataset Collector'
PG_TEXT_ANTIALIAS = True

# joystick constants:

DEFAULT_JS1_PITCH_AX = 0
DEFAULT_JS1_ROLL_AX = 1
DEFAULT_JS1_YAW_AX = 2
DEFAULT_JS1_Z_AX = 3

DEFAULT_JS2_PITCH_AX = 0
DEFAULT_JS2_ROLL_AX = 1
DEFAULT_JS2_YAW_AX = 2
DEFAULT_JS2_Z_AX = 3

DEFAULT_JS1_PITCH_GAIN = 0.5
DEFAULT_JS1_ROLL_GAIN = 0.5
DEFAULT_JS1_YAW_GAIN = -2.0
DEFAULT_JS1_Z_GAIN = -0.5

DEFAULT_JS2_PITCH_GAIN = 0.5
DEFAULT_JS2_ROLL_GAIN = 0.5
DEFAULT_JS2_YAW_GAIN = -2.0
DEFAULT_JS2_Z_GAIN = -0.5

JS_CONFIG_TIME = 5.0

JOYSTICKS_SETTINGS_FILE = PG_GAME_FOLDER_PATH / "js.obj"

# airsim constants:

DRONE1_NAME = 'drone_1'  # this is the name of the first drone in the .json file
DRONE2_NAME = 'drone_2'  # this is the name of the second drone in the .json file
DEFAULT_DRONE1_OFFSET = airsim.Vector3r(0.0, 0.0, 0.0)  # this is the start position of the first drone in the .json file
DEFAULT_DRONE2_OFFSET = airsim.Vector3r(0.0, 0.0, 0.0)  # this is the start position of the second drone in the .json file
DISPLAY_MODE = "NO_DISPLAY"  # setting this to "WINDOWED_DISPLAY" will launch the arisim display alongside the game

# environment constants:
print(SYS_STR)

TIME_STEP_METHOD = 'CONTINUE_FOR_TIME' if SYS_STR == 'Linux' else 'JOIN'  # CONTINUE_FOR_TIME seems to break rendering on Windows
IMG_OBS_NAME = 'front_camera_dc'
DEFAULT_CONTROL_API = 'moveByRollPitchYawrateZAsync'
DEFAULT_OBS_COORD = "all"  # 'dc', 'ned', 'global' or 'all'
DEFAULT_ACT_COORD = "ned"  # 'dc', 'ned' or 'global' (!: check API support in the environment before changing this)
DEFAULT_RF_CONFIG = {  # parameters of the reward function
    'constant_penalty': -1.0,  # constant penalty per time-step
    'collision_radius': 0.5,  # at this distance from the opponen, the lagging drone dies (don't change for now, this is enforced by the airsim .pak file)
    'velocity_gain': 10.0,  # not real velocity: difference of distance to next objective between 2 get_reward()
    'gate_crossed_reward': 100.0,
    'gate_missed_penalty': -100.0,
    'collision_penatly': -10,  # collision with environment
    'death_penalty': -500,  # collision with opponent
    'death_constant_penalty': 0.0,  # after collision with opponent until the end of track (should be at least lower than constant summed penalty when lagging behind and not moving to avoid reward hacking)
    'end_of_track_bonus': 100.0,  # only when the last gate is crossed
    'lag_penalty': -0.5,  # constant additional penalty if not leading the way
    'kill_reward': 50.0,
    'gate_facing_reward_gain': 1.0
}
DEFAULT_LEVEL = "Building99_Hard"
DEFAULT_TIER = 0
DEFAULT_CLOCK_SPEED = 1.0
DEFAULT_IMG_WIDTH = 320
DEFAULT_IMG_HEIGHT = 240
DEFAULT_TIME_STEP_DURATION = 0.1
DEFAULT_EPISODE_DURATION = 100.0
DEFAULT_TIME_STOP = True
DEFAULT_REAL_TIME = False
DEFAULT_ACT_THREADING = False

Z_TARGETS_START = {"Building99_Hard": 1.0,
                   "Soccer_Field_Easy": 0.0,
                   "Soccer_Field_Medium": 0.0,
                   "ZhangJiaJie_Medium": 3.0}

# keyboard constants:

FORWARD1 = pg.K_y
BACKWARD1 = pg.K_h
RIGHT1 = pg.K_u
LEFT1 = pg.K_t
RIGHTYAW1 = pg.K_j
LEFTYAW1 = pg.K_g
UP1 = pg.K_o
DOWN1 = pg.K_l
FORWARD2 = pg.K_KP8
BACKWARD2 = pg.K_KP5
RIGHT2 = pg.K_KP9
LEFT2 = pg.K_KP7
RIGHTYAW2 = pg.K_KP6
LEFTYAW2 = pg.K_KP4
UP2 = pg.K_PAGEUP
DOWN2 = pg.K_PAGEDOWN
EXIT = pg.K_ESCAPE  # exit game
RESET = pg.K_q  # discards and resets episode
SAVE_EPISODES = pg.K_s  # toggles dataset reccording
CONFIG_JS = pg.K_c  # calls joystick configuration

# dataset constants:

DATASET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'drone_racing_dataset_collector', 'dataset')
DEFAULT_PLAYER_1 = 'player_1'
DEFAULT_PLAYER_2 = 'player_2'

# print options:

CHAR_PER_OBS = 40
PRINT_TRANSITIONS = False  # True for debugging (prints all the observations in terminal)

# others:

DFAULT_PROFILER = False  # True to profile the code with PyInstrument by default


# these function are only for printing transitions in terminal:

def quaternion_to_euler(q):
    x_val = q[0]
    y_val = q[1]
    z_val = q[2]
    w_val = q[3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w_val * x_val + y_val * z_val)
    cosr_cosp = 1 - 2 * (x_val * x_val + y_val * y_val)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2 * (w_val * y_val - z_val * x_val)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w_val * z_val + x_val * y_val)
    cosy_cosp = 1 - 2 * (y_val * y_val + z_val * z_val)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def print_obs_n(obs_n, rew_n, done_n, convert_quat=True):
    """
    !: this will only work with less than 9 agents
    if convert_quat==True, this shows the corresponding euler angles of the quaternions
    """
    d = {}
    nb_agents = len(obs_n)
    for agent in range(nb_agents):
        for key, item in obs_n[agent].items():
            words = key.split('_')
            letters = [word[0] for word in words] + [f'{agent}']
            new_key = "".join(letters)
            if not convert_quat:
                d[new_key] = item
            else:
                d[new_key] = quaternion_to_euler(item) if "orientation" in key else item
    for agent in range(nb_agents):
        d[f"_r{agent}"] = np.array([rew_n[agent]])
        d[f"_d{agent}"] = np.array([done_n[agent]])
    lks = list(d.keys())
    lks.sort()
    print("---")
    for k in lks:
        if 'fc' in k:  # front_camera
            str = f"{k}:{d[k].shape}"
        else:
            str = f"{k}:{d[k]}"
        idx = int(k[-1])
        end = "\n" if idx == nb_agents - 1 else " " * (CHAR_PER_OBS - len(str))
        print(str, end=end)


# These function handle joystick configuration:

class JoysticksSettings(object):
    def __init__(self):
        self.JS1_PITCH_GAIN = DEFAULT_JS1_PITCH_GAIN
        self.JS1_ROLL_GAIN = DEFAULT_JS1_ROLL_GAIN
        self.JS1_YAW_GAIN = DEFAULT_JS1_YAW_GAIN
        self.JS1_Z_GAIN = DEFAULT_JS1_Z_GAIN

        self.JS1_PITCH_AX = DEFAULT_JS1_PITCH_AX
        self.JS1_ROLL_AX = DEFAULT_JS1_ROLL_AX
        self.JS1_YAW_AX = DEFAULT_JS1_YAW_AX
        self.JS1_Z_AX = DEFAULT_JS1_Z_AX

        self.JS2_PITCH_GAIN = DEFAULT_JS2_PITCH_GAIN
        self.JS2_ROLL_GAIN = DEFAULT_JS2_ROLL_GAIN
        self.JS2_YAW_GAIN = DEFAULT_JS2_YAW_GAIN
        self.JS2_Z_GAIN = DEFAULT_JS2_Z_GAIN

        self.JS2_PITCH_AX = DEFAULT_JS2_PITCH_AX
        self.JS2_ROLL_AX = DEFAULT_JS2_ROLL_AX
        self.JS2_YAW_AX = DEFAULT_JS2_YAW_AX
        self.JS2_Z_AX = DEFAULT_JS2_Z_AX


def dump_joysticks_settings(jset):
    with open(JOYSTICKS_SETTINGS_FILE, 'wb') as f:
        pickle.dump(jset, f)


def load_joysticks_settings():
    if JOYSTICKS_SETTINGS_FILE.exists():
        print(f"Loading joystick settings from {JOYSTICKS_SETTINGS_FILE}")
        with open(JOYSTICKS_SETTINGS_FILE, 'rb') as f:
            jset = pickle.load(f)
    else:
        print(f"No setting file found. Creating a new one at {JOYSTICKS_SETTINGS_FILE}")
        jset = JoysticksSettings()
        dump_joysticks_settings(jset)
    return jset


def configure_joysticks_settings(jset, js, eps=0.05):
    for i, j in enumerate(js):
        na = j.get_numaxes()
        axvals_neutral = np.array([0.0, ] * na)
        axvals = np.array([0.0, ] * na)
        axvals_dif = np.array([0.0, ] * na)
        print(f"--- Now configuring joystick {i} ---")
        print(f"Number of detected axis: {na}")
        time.sleep(JS_CONFIG_TIME)
        print(f"JOYSTICK {i}: Leave all the axis untouched:")
        time.sleep(JS_CONFIG_TIME)
        for _ in range(10):
            time.sleep(JS_CONFIG_TIME / 10)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    break
            for ax in range(na):
                axvals_neutral[ax] = j.get_axis(ax)
            print(f"JOYSTICK {i}: detected values: {axvals_neutral}")
        print(f"JOYSTICK {i}: neutral values: {axvals_neutral}")
        time.sleep(JS_CONFIG_TIME)

        print(f"JOYSTICK {i}: HOLD FULL PITCH (forward):")
        time.sleep(JS_CONFIG_TIME)
        for _ in range(10):
            time.sleep(JS_CONFIG_TIME / 10)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    break
            for ax in range(na):
                axvals[ax] = j.get_axis(ax)
            print(f"JOYSTICK {i}: detected values: {axvals}")
        axvals_dif = axvals - axvals_neutral
        argmax_ax = np.argmax(np.abs(axvals_dif))
        delta = axvals_dif[argmax_ax]
        print(f"JOYSTICK {i}: selected PITCH axis:{argmax_ax} with a delta of {delta}")
        if np.abs(delta) <= eps:
            print("ERROR: delta too small.")
            time.sleep(JS_CONFIG_TIME)
            break
        if i == 0:
            jset.JS1_PITCH_AX = argmax_ax
            jset.JS1_PITCH_GAIN = np.sign(delta) * DEFAULT_JS1_PITCH_GAIN
        else:
            jset.JS2_PITCH_AX = argmax_ax
            jset.JS2_PITCH_GAIN = np.sign(delta) * DEFAULT_JS2_PITCH_GAIN
        time.sleep(JS_CONFIG_TIME)

        print(f"JOYSTICK {i}: HOLD FULL THROTTLE (up) for {JS_CONFIG_TIME} seconds:")
        time.sleep(JS_CONFIG_TIME)
        for _ in range(10):
            time.sleep(JS_CONFIG_TIME / 10)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    break
            for ax in range(na):
                axvals[ax] = j.get_axis(ax)
            print(f"JOYSTICK {i}: detected values: {axvals}")
        axvals_dif = axvals - axvals_neutral
        argmax_ax = np.argmax(np.abs(axvals_dif))
        delta = axvals_dif[argmax_ax]
        print(f"JOYSTICK {i}: selected THROTTLE axis:{argmax_ax} with a delta of {delta}")
        if np.abs(delta) <= eps:
            print("ERROR: delta too small.")
            time.sleep(JS_CONFIG_TIME)
            break
        if i == 0:
            jset.JS1_Z_AX = argmax_ax
            jset.JS1_Z_GAIN = np.sign(delta) * DEFAULT_JS1_Z_GAIN
        else:
            jset.JS2_Z_AX = argmax_ax
            jset.JS2_Z_GAIN = np.sign(delta) * DEFAULT_JS2_Z_GAIN
        time.sleep(JS_CONFIG_TIME)

        print(f"JOYSTICK {i}: HOLD FULL ROLL (right) for {JS_CONFIG_TIME} seconds:")
        time.sleep(JS_CONFIG_TIME)
        for _ in range(10):
            time.sleep(JS_CONFIG_TIME / 10)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    break
            for ax in range(na):
                axvals[ax] = j.get_axis(ax)
            print(f"JOYSTICK {i}: detected values: {axvals}")
        axvals_dif = axvals - axvals_neutral
        argmax_ax = np.argmax(np.abs(axvals_dif))
        delta = axvals_dif[argmax_ax]
        print(f"JOYSTICK {i}: selected ROLL axis:{argmax_ax} with a delta of {delta}")
        if np.abs(delta) <= eps:
            print("ERROR: delta too small.")
            time.sleep(JS_CONFIG_TIME)
            break
        if i == 0:
            jset.JS1_ROLL_AX = argmax_ax
            jset.JS1_ROLL_GAIN = np.sign(delta) * DEFAULT_JS1_ROLL_GAIN
        else:
            jset.JS2_ROLL_AX = argmax_ax
            jset.JS2_ROLL_GAIN = np.sign(delta) * DEFAULT_JS2_ROLL_GAIN
        time.sleep(JS_CONFIG_TIME)

        print(f"JOYSTICK {i}: HOLD FULL YAW (look right) for {JS_CONFIG_TIME} seconds:")
        time.sleep(JS_CONFIG_TIME)
        for _ in range(10):
            time.sleep(JS_CONFIG_TIME / 10)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    break
            for ax in range(na):
                axvals[ax] = j.get_axis(ax)
            print(f"JOYSTICK {i}: detected values: {axvals}")
        axvals_dif = axvals - axvals_neutral
        argmax_ax = np.argmax(np.abs(axvals_dif))
        delta = axvals_dif[argmax_ax]
        print(f"JOYSTICK {i}: selected YAW axis:{argmax_ax} with a delta of {delta}")
        if np.abs(delta) <= eps:
            print("ERROR: delta too small.")
            time.sleep(JS_CONFIG_TIME)
            break
        if i == 0:
            jset.JS1_YAW_AX = argmax_ax
            jset.JS1_YAW_GAIN = np.sign(delta) * DEFAULT_JS1_YAW_GAIN
        else:
            jset.JS2_YAW_AX = argmax_ax
            jset.JS2_YAW_GAIN = np.sign(delta) * DEFAULT_JS2_YAW_GAIN
        time.sleep(JS_CONFIG_TIME)
    dump_joysticks_settings(jset)
    return jset


class GameOfDatasets(object):
    def __init__(self,
                 tier,
                 clock_speed,
                 simulated_time_step,
                 ep_max_length,
                 level_name,
                 img_width,
                 img_height,
                 control_method_str,
                 act_coord_system,
                 obs_coord_system,
                 rf_config,
                 time_stop,
                 real_time,
                 act_threading,
                 experiment_name,
                 player_1,
                 player_2):
        self.tier = tier
        self.clock_speed = clock_speed
        self.simulated_time_step = simulated_time_step
        self.ep_max_length = ep_max_length
        self.level_name = level_name
        self.img_width = img_width
        self.img_height = img_height
        self.control_method_str = control_method_str
        self.act_coord_system = act_coord_system
        self.obs_coord_system = obs_coord_system
        self.rf_config = rf_config
        self.experiment_name = experiment_name
        self.players = [player_1, player_2]
        self.episode_num = 0
        self.save_ep = False

        # pygame init:

        pg.init()
        pg.joystick.init()
        self.nb_js = pg.joystick.get_count()
        print(f"INFO: {self.nb_js} connected joysticks detected")
        self.js = [pg.joystick.Joystick(i) for i in range(self.nb_js)]
        for i, j in enumerate(self.js):
            j.init()
            na = j.get_numaxes()
            print(f"INFO: joystick {i}: {na} axes")
            assert na >= 4, "the connected joystick is not compatible"
        self.txt_height = 50
        self.info_height = 50
        self.font1 = pg.font.Font('freesansbold.ttf', 12)
        self.display_width = max(PG_DISPLAY_WIDTH, 2 * self.img_width)
        self.display_height = max(PG_DISPLAY_HEIGHT, self.img_height + self.txt_height + self.info_height)
        self.gameDisplay = pg.display.set_mode((self.display_width, self.display_height))
        pg.display.set_caption(PG_GAME_CAPTION)
        welcomeImg = pg.image.load(str(PG_GAME_FOLDER_PATH / 'sprites/welcome.jpg'))
        self.gameDisplay.blit(welcomeImg, (0, 0))
        pg.display.update()

        # environment init:

        # actions bounds:
        low_bounds = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
        high_bounds = np.array([np.inf, np.inf, np.inf, np.inf])

        if(tier == 2):
            DRONES_NAMES = [DRONE1_NAME]
            DRONES_OFFSETS = [DEFAULT_DRONE1_OFFSET]
        else:
            DRONES_NAMES = [DRONE1_NAME, DRONE2_NAME]
            DRONES_OFFSETS = [DEFAULT_DRONE1_OFFSET, DEFAULT_DRONE2_OFFSET]

        config = {
            "airsim_client": None,
            "clock_speed": clock_speed,
            "control_method_str": control_method_str,
            "low_bounds": low_bounds,
            "high_bounds": high_bounds,
            "simulated_time_step": simulated_time_step,
            "drones_names": DRONES_NAMES,
            "drones_offsets": DRONES_OFFSETS,
            "level_name": level_name,
            "tier": tier,
            "ep_max_length": ep_max_length,
            "dummy": False,
            "history_length": 1,
            "img_width": img_width,
            "img_height": img_height,
            "act_coord_system": act_coord_system,
            "obs_coord_system": obs_coord_system,
            "rf_config": rf_config,
            "use_locker": False,
            "time_step_method": TIME_STEP_METHOD,
            "rendering_mode": DISPLAY_MODE,
            "time_stop": time_stop,
            "real_time": real_time,
            "act_threading": act_threading,
            "default_z_target": Z_TARGETS_START[level_name] if level_name in Z_TARGETS_START else 0.0
        }

        self.env = gym.make('gym_game_of_drones.envs:gym-airsimdroneracinglab-v0', config=config)
        self.obs_n = self.env.reset()
        self.rew_n = None
        self.done_n = None
        self.info_n = None
        self.is_img_obs = IMG_OBS_NAME in self.obs_n[0]
        self.nb_players = len(self.obs_n)
        self.steps = deque(maxlen=self.ep_max_length + 1)
        self.cum_rew_n = [0.0] * self.nb_players
        self.img_surfaces = []
        self.txt_surfaces = []
        self.info_surface = pg.Surface((self.display_width, self.info_height))
        for p in range(self.nb_players):
            self.img_surfaces.append(pg.Surface((self.img_width, self.img_height)))
            self.txt_surfaces.append(pg.Surface((self.img_width, self.txt_height)))
        self.background_set = False
        self.x_img_disp_subsurf = self.display_width / self.nb_players
        self.x_img_disp_offset = (self.x_img_disp_subsurf - self.img_width) / 2.0
        self.y_img_disp_offset = (self.display_height - self.img_height + self.txt_height) / 2.0
        self.y_txt_disp_offset = (self.display_height - self.img_height - self.txt_height) / 2.0
        self.y_info_disp_offset = 10
        self.start_time = time.time()
        self.cur_dur = time.time() - self.start_time

    def __del__(self):
        self.env.stop()
        for j in self.js:
            j.quit()
        pg.joystick.quit()
        pg.quit()

    def reset(self):
        self.obs_n = self.env.reset()
        self.rew_n = None
        self.cum_rew_n = [0.0] * self.nb_players
        self.done_n = [False] * self.nb_players
        self.info_n = None
        self.steps.clear()
        self.steps.append({'act_n': None,
                           'rew_n': None,
                           'obs_n': copy.deepcopy(self.obs_n),
                           'done_n': copy.deepcopy(self.done_n)})
        self.start_time = time.time()
        self.cur_dur = time.time() - self.start_time

    def save_episode(self):
        dir_path = os.path.join(DATASET_DIR, f'{self.experiment_name}_{self.episode_num}.pkl')
        with open(dir_path, 'wb') as output:
            pickle.dump(self.steps, output, pickle.HIGHEST_PROTOCOL)
        self.episode_num = self.episode_num + 1

    def display(self):
        if not self.background_set:
            self.gameDisplay.fill(PG_BACKGROUND_COLOR)
            self.background_set = True
        cur_step=self.env.current_step
        textcs = self.font1.render(f"time step : {cur_step} / 1000", PG_TEXT_ANTIALIAS, PG_TEXT_COLOR, PG_BACKGROUND_COLOR)
        textrRectcs = textcs.get_rect()
        textrRectcs.center = (self.display_width / 2.0, 1.0 * self.info_height / 3.0)
        textct = self.font1.render(f"time : {self.cur_dur:.1f} / 300", PG_TEXT_ANTIALIAS, PG_TEXT_COLOR, PG_BACKGROUND_COLOR)
        textrRectct = textct.get_rect()
        textrRectct.center = (self.display_width / 2.0, 2.0 * self.info_height / 3.0)
        
        self.info_surface.fill(PG_BACKGROUND_COLOR)
        self.info_surface.blit(textcs, textrRectcs)
        self.info_surface.blit(textct, textrRectct)
        self.gameDisplay.blit(self.info_surface, (0, self.y_info_disp_offset))
        for p in range(self.nb_players):
            if self.is_img_obs:
                # print(f"DEBUG: img shape: {self.obs_n[p][IMG_OBS_NAME][0].shape}")
                pg.surfarray.blit_array(self.img_surfaces[p], np.swapaxes(self.obs_n[p][IMG_OBS_NAME][0], 0, 1))
            else:
                self.img_surfaces[p].fill(PG_IMG_BACKGROUND_COLOR)
            x = self.x_img_disp_subsurf * p + self.x_img_disp_offset
            y = self.y_img_disp_offset
            self.gameDisplay.blit(self.img_surfaces[p], (x, y))
            rew = self.rew_n[p] if self.rew_n else 0.0
            crew = self.cum_rew_n[p] if self.rew_n else 0.0
            plyr = self.players[p] 
            textp = self.font1.render(f"{plyr}", PG_TEXT_ANTIALIAS, PG_TEXT_COLOR, PG_BACKGROUND_COLOR)
            textrRectp = textp.get_rect()
            textrRectp.center = (self.img_width / 2.0, 1.0 * self.txt_height / 3.0)
            textr = self.font1.render(f"cr: {crew:.2f} (r: {rew:.2f})", PG_TEXT_ANTIALIAS, PG_TEXT_COLOR, PG_BACKGROUND_COLOR)
            textrRect = textr.get_rect()
            textrRect.center = (self.img_width / 2.0, 2.0 * self.txt_height / 3.0)
            self.txt_surfaces[p].fill(PG_BACKGROUND_COLOR)
            self.txt_surfaces[p].blit(textr, textrRect)
            self.txt_surfaces[p].blit(textp, textrRectp)
            self.gameDisplay.blit(self.txt_surfaces[p], (x, self.y_txt_disp_offset))
        pg.display.update()

    def run_game(self):
        pitch1 = 0.0
        pitch2 = 0.0
        roll1 = 0.0
        roll2 = 0.0
        yaw_rate1 = 0.0
        yaw_rate2 = 0.0
        z1 = 0.0
        z2 = 0.0

        # joysticks settings:

        jset = load_joysticks_settings()

        # main loop:

        c = True
        while(c):

            # keyboard controller:

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    c = False

            keys = pg.key.get_pressed()

            if keys[EXIT]:
                c = False

            if keys[CONFIG_JS]:
                jset = configure_joysticks_settings(jset, self.js)

            if self.nb_js == 0:
                if keys[FORWARD1]:
                    pitch1 = 0.1
                elif keys[BACKWARD1]:
                    pitch1 = -0.1
                else:
                    pitch1 = 0.0

                if keys[RIGHT1]:
                    roll1 = 0.1
                elif keys[LEFT1]:
                    roll1 = -0.1
                else:
                    roll1 = 0.0

                if keys[RIGHTYAW1]:
                    yaw_rate1 = -0.5
                elif keys[LEFTYAW1]:
                    yaw_rate1 = 0.5
                else:
                    yaw_rate1 = 0.0

                if keys[UP1]:
                    z1 = -0.1
                elif keys[DOWN1]:
                    z1 = 0.1
                else:
                    z1 = 0.0

            if self.nb_js <= 1:
                if keys[FORWARD2]:
                    pitch2 = 0.1
                elif keys[BACKWARD2]:
                    pitch2 = -0.1
                else:
                    pitch2 = 0.0

                if keys[RIGHT2]:
                    roll2 = 0.1
                elif keys[LEFT2]:
                    roll2 = -0.1
                else:
                    roll2 = 0.0

                if keys[RIGHTYAW2]:
                    yaw_rate2 = -0.5
                elif keys[LEFTYAW2]:
                    yaw_rate2 = 0.5
                else:
                    yaw_rate2 = 0.0

                if keys[UP2]:
                    z2 = -0.1
                elif keys[DOWN2]:
                    z2 = 0.1
                else:
                    z2 = 0.0

            for i, j in enumerate(self.js):
                if i == 0:
                    pitch1 = j.get_axis(jset.JS1_PITCH_AX) * jset.JS1_PITCH_GAIN
                    roll1 = j.get_axis(jset.JS1_ROLL_AX) * jset.JS1_ROLL_GAIN
                    yaw_rate1 = j.get_axis(jset.JS1_YAW_AX) * jset.JS1_YAW_GAIN
                    z1 = j.get_axis(jset.JS1_Z_AX) * jset.JS1_Z_GAIN
                elif i == 1:
                    pitch2 = j.get_axis(jset.JS2_PITCH_AX) * jset.JS2_PITCH_GAIN
                    roll2 = j.get_axis(jset.JS2_ROLL_AX) * jset.JS2_ROLL_GAIN
                    yaw_rate2 = j.get_axis(jset.JS2_YAW_AX) * jset.JS2_YAW_GAIN
                    z2 = j.get_axis(jset.JS2_Z_AX) * jset.JS2_Z_GAIN

            if self.tier == 2:
                action_n = np.array([[roll1, pitch1, yaw_rate1, z1]])
            else:
                action_n = np.array([[roll1, pitch1, yaw_rate1, z1],
                                    [roll2, pitch2, yaw_rate2, z2]])

            self.obs_n, self.rew_n, self.done_n, self.info_n = self.env.step(action_n)
            self.steps.append({'act_n': copy.deepcopy(action_n),
                               'rew_n': copy.deepcopy(self.rew_n),
                               'obs_n': copy.deepcopy(self.obs_n),
                               'done_n': copy.deepcopy(self.done_n)})
            for p, r in enumerate(self.rew_n):
                self.cum_rew_n[p] = self.cum_rew_n[p] + r

            if PRINT_TRANSITIONS:
                print_obs_n(self.obs_n, self.rew_n, self.done_n)
            self.display()

            all_done = True
            for done in self.done_n:
                if not done:
                    all_done = False
            
            self.cur_dur = time.time() - self.start_time
            if self.cur_dur >= 300.0:
                print("WARNING: 300 seconds elapsed (probably your computer is slow at rendering images), which is not compatible with the current racing binaries: discarding episode...")
                self.reset()
            elif keys[RESET]:
                print("Manual reset, discarding episode...")
                self.reset()
            elif keys[SAVE_EPISODES]:
                self.save_ep = not self.save_ep
                print(f"Save future episodes: {self.save_ep}")
                self.reset()
            elif all_done:
                print("All done!")
                if self.save_ep:
                    print("Saving episode...")
                    self.save_episode()
                self.reset()


def main(args):
    clock_speed = args.clock_speed  # this is the clock-speed value in the .json file
    simulated_time_step = args.simulated_time_step
    episode_max_duration = args.episode_max_duration
    control_method_str = args.control_method_str
    act_coord_system = args.act_coord_system
    obs_coord_system = args.obs_coord_system
    rf_config = args.rf_config
    # CPU_TIME_STEP = TIME_STEP / SIMULATOR_SPEED  # this is the actual CPU time step
    NB_TIME_STEPS_EPISODE = round(episode_max_duration / simulated_time_step)
    level_name = args.level_name
    tier = args.tier
    img_width = args.img_width
    img_height = args.img_height
    time_stop = args.time_stop
    real_time = args.real_time
    act_threading = args.act_threading
    player_1 = args.player_1
    player_2 = args.player_2
    experiment_name = args.experiment_name
    experiment_name = f"{experiment_name}_{player_1}_{player_2}"
    benchmark = args.benchmark

    god = GameOfDatasets(tier=tier,
                         clock_speed=clock_speed,
                         simulated_time_step=simulated_time_step,
                         ep_max_length=NB_TIME_STEPS_EPISODE,
                         level_name=level_name,
                         img_width=img_width,
                         img_height=img_height,
                         control_method_str=control_method_str,
                         act_coord_system=act_coord_system,
                         obs_coord_system=obs_coord_system,
                         rf_config=rf_config,
                         time_stop=time_stop,
                         real_time=real_time,
                         act_threading=act_threading,
                         experiment_name=experiment_name,
                         player_1=player_1,
                         player_2=player_2)
    if benchmark:
        pro = Profiler()
        pro.start()
    god.run_game()
    if benchmark:
        pro.stop()
        print(pro.output_text(unicode=True, color=False))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--level_name', type=str, choices=["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard"], default=DEFAULT_LEVEL)
    parser.add_argument('--player_1', type=str, default=DEFAULT_PLAYER_1)
    parser.add_argument('--player_2', type=str, default=DEFAULT_PLAYER_2)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--tier', type=int, choices=[0, 1, 2, 3], default=DEFAULT_TIER)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=41451)
    parser.add_argument('--clock_speed', type=float, default=DEFAULT_CLOCK_SPEED)
    parser.add_argument('--img_width', type=int, default=DEFAULT_IMG_WIDTH)
    parser.add_argument('--img_height', type=int, default=DEFAULT_IMG_HEIGHT)
    parser.add_argument('--simulated_time_step', type=float, default=DEFAULT_TIME_STEP_DURATION)
    parser.add_argument('--episode_max_duration', type=float, default=DEFAULT_EPISODE_DURATION)
    parser.add_argument('--control_method_str', type=str, default=DEFAULT_CONTROL_API)
    parser.add_argument('--act_coord_system', type=str, default=DEFAULT_ACT_COORD)
    parser.add_argument('--obs_coord_system', type=str, default=DEFAULT_OBS_COORD)
    parser.add_argument('--rf_config', type=dict, default=DEFAULT_RF_CONFIG)
    parser.add_argument('--time_stop', dest='time_stop', action='store_true')
    parser.add_argument('--no-time_stop', dest='time_stop', action='store_false')
    parser.set_defaults(time_stop=DEFAULT_TIME_STOP)
    parser.add_argument('--real_time', dest='real_time', action='store_true')
    parser.add_argument('--no-real_time', dest='real_time', action='store_false')
    parser.set_defaults(real_time=DEFAULT_REAL_TIME)
    parser.add_argument('--act_threading', dest='act_threading', action='store_true')
    parser.add_argument('--no-act_threading', dest='act_threading', action='store_false')
    parser.set_defaults(act_threading=DEFAULT_ACT_THREADING)
    parser.add_argument('--benchmark', dest='benchmark', action='store_true')
    parser.add_argument('--no-benchmark', dest='benchmark', action='store_false')
    parser.set_defaults(act_threading=DFAULT_PROFILER)
    args = parser.parse_args()
    main(args)
