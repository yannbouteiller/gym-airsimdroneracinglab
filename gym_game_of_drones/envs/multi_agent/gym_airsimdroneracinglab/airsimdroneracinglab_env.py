# This is the gym environment core
# Several flavors are available
# In some flavors, actions are taken at the NEXT time-step (Real Time RL setting)
# In some flavors, actions are taken at the CURRENT time-step (usual RL setting)
# Some flavors pause the environment between each step (time stop)
# Some flavors run continuously (no time control) - for these flavors step() has to be called from a timer
# Depending on the values of syncronous_actions and synchronous_states, the simulator can be paused or not to retrieve observations for both drones and send them new actions
# To look into how rewards are defined, read rewardfunction.py

from gym import Env
import gym.spaces as spaces
import numpy as np
from gym_game_of_drones.envs.multi_agent.gym_airsimdroneracinglab import rewardfunction as rf
from gym_game_of_drones.envs.multi_agent.gym_airsimdroneracinglab.custom_airsim_settings_creator import CustomAirSimSettingsCreator
import time
import random
import airsimdroneracinglab as airsim
import subprocess
import os
import signal
import pickle
import copy
from gym_game_of_drones.envs.multi_agent.gym_airsimdroneracinglab.rllib_compatibility_client import LockerClient
from collections import deque
from threading import Thread
from platform import system
from pathlib import Path

DEFAULT_IP_PORT_FILE_NAME = 'ip_port.obj'

SYS_STR = system()
if SYS_STR == 'Linux':
    SUB_DIR = 'ADRL/ADRL/Binaries/Linux'
    EXECUTABLE_NAME = 'ADRL'
    OPTIONS_DISPLAY = '-windowed -opengl4'  # used with default DISPLAY
    OPTIONS_WINDOWED_NO_DISPLAY = '-windowed -opengl4 -BENCHMARK'
    OPTIONS_NO_DISPLAY = '-opengl4 -BENCHMARK'
    OPTIONS_NO_RENDER = '-nullrhi'  # used with DISPLAY=""
else:
    SUB_DIR = 'ADRL/ADRL/Binaries/Win64'
    EXECUTABLE_NAME = 'ADRL.exe'
    OPTIONS_DISPLAY = ['-windowed']  # used with default DISPLAY
    OPTIONS_WINDOWED_NO_DISPLAY = ['-windowed']
    OPTIONS_NO_DISPLAY = ['-windowed']
    OPTIONS_NO_RENDER = ['-nullrhi']  # used with DISPLAY=""

DEFAULT_RENDERING_MODE = 'NO_DISPLAY'  # 'WINDOWED_DISPLAY', 'WINDOWED_NO_DISPLAY', 'NO_DISPLAY' or 'NO_RENDER'
RESET_TIMEOUT = 10  # seconds before we consider the simulator froze and needs to be killed and launched again

DEFAULT_IMG_HEIGHT = 240
DEFAULT_IMG_WIDTH = 320

MAX_GETIMAGES_TRIALS = 100
SLEEP_TIME_AT_RESETRACE = 0.1

DEFAULT_RF_CONFIG = {
    'constant_penalty': -1.0,  # constant penalty per time-step
    'collision_radius': 0.5,  # collision with opponent
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


class IpPort(object):
    def __init__(self,
                 global_ip_count_1=1,
                 global_ip_count_2=0,
                 global_ip_count_3=0,
                 global_ip_count_4=127,
                 global_port_count=41451):
        self.global_ip_count_1 = global_ip_count_1
        self.global_ip_count_2 = global_ip_count_2
        self.global_ip_count_3 = global_ip_count_3
        self.global_ip_count_4 = global_ip_count_4
        self.global_port_count = global_port_count


def print_with_pid(*args):
    """
    helper function: prints along with the process pid for rllib debugging
    """
    pid = os.getpid()
    print('(', pid, ') ', *args)


def initialize_ip_port_file(ip_port_file_name=DEFAULT_IP_PORT_FILE_NAME):
    """
    This should be called prior to creating the first environment
    """
    ip_port = IpPort()
    f = open(ip_port_file_name, 'wb')
    pickle.dump(ip_port, f)
    f.close()


def new_client(clockspeed, img_height, img_width, ip_port_file_name=DEFAULT_IP_PORT_FILE_NAME, mode='WINDOWED_DISPLAY', use_locker=True, lock_client=None):
    """
    This function safely creates a new airsim client
    It can be used to create several clients in parallel
    a lock server has to be created by the calling process with rllib_compatibility.init_rllib_compatibility_server
    """
    if use_locker:
        print_with_pid("DEBUG: new_client: waiting for lock...")
        lock_client.acquire()
        print_with_pid("DEBUG: new_client: lock acquired")
    # else:
    #     rand_time = np.random.random() * 100
    #     print_with_pid(f"DEBUG: sleeping for {rand_time} s before starting")
    #     time.sleep(rand_time)
    #     print_with_pid(f"DEBUG: stopped sleeping")

    if not Path(ip_port_file_name).is_file():
        initialize_ip_port_file(ip_port_file_name)
    f = open(ip_port_file_name, 'rb')
    ip_port = pickle.load(f)
    f.close()

    port = ip_port.global_port_count
    ip_str = f"{ip_port.global_ip_count_4}.{ip_port.global_ip_count_3}.{ip_port.global_ip_count_2}.{ip_port.global_ip_count_1}"
    print_with_pid("DEBUG: ip:", ip_str, ", port:", port)

    ip_port.global_port_count += 1
    if ip_port.global_port_count >= 65535:
        ip_port.global_port_count = 41451
        ip_port.global_ip_count_1 += 1
        if ip_port.global_ip_count_1 >= 256:
            ip_port.global_ip_count_2 += 1
            ip_port.global_ip_count_1 = 0
    assert ip_port.global_ip_count_2 < 256, "ERROR: too many environments have ben created, IP overflow"

    f = open(ip_port_file_name, 'wb')
    pickle.dump(ip_port, f)
    f.close()

    my_env = os.environ.copy()
    # print_with_pid(f"DEBUG: setting SDL_HINT_CUDA_DEVICE to {gpu_str}")
    my_env["SDL_HINT_CUDA_DEVICE"] = '0'
    if mode == 'NO_RENDER':  # for tier 1
        options = OPTIONS_NO_RENDER
        viewmode = "NoDisplay"
        my_env["DISPLAY"] = ""
    elif mode == 'NO_DISPLAY':  # for tier 2, 3
        options = OPTIONS_NO_DISPLAY
        viewmode = "NoDisplay"
        my_env["DISPLAY"] = ""
    elif mode == 'WINDOWED_NO_DISPLAY':
        options = OPTIONS_WINDOWED_NO_DISPLAY
        viewmode = "NoDisplay"
    else:
        print("DEBUG: mode = default")
        options = OPTIONS_DISPLAY
        viewmode = "FlyWithMe"

    CustomAirSimSettingsCreator().write_custom_settings_file(clockspeed=clockspeed, ip=ip_str, port=port, img_height=img_height, img_width=img_width, viewmode=viewmode)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, SUB_DIR, EXECUTABLE_NAME)

    if SYS_STR == 'Linux':
        p = subprocess.Popen(dir_path + ' ' + options, bufsize=-1, stdout=None, shell=True, preexec_fn=os.setpgrp, env=my_env)
    else:
        p = subprocess.Popen([dir_path] + options, bufsize=-1, stdout=None, shell=False, env=my_env)
        # subprocess.run([dir_path, options], capture_output=False, shell=False, env=my_env)
        # p = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

    print_with_pid("DEBUG: waiting for AirSim to start...")
    time.sleep(10)  # this is to wait for AirSim to load fully
    print_with_pid("DEBUG: stopped waiting")
    airsim_client = airsim.MultirotorClient(ip=ip_str, port=port, timeout_value=RESET_TIMEOUT)

    print_with_pid("DEBUG: confirmConnection()...")
    airsim_client.confirmConnection()
    time.sleep(2.0)

    airsim_client.simDisableRaceLog()

    if use_locker:
        print_with_pid("DEBUG: new_client: releasing lock")
        lock_client.release()

    return airsim_client, p


def basis_oriented_vector(drone_base_orientation, base_vector):
    """
    performs a change of basis for airsim vectors
    caution: this only *rotates* base_vector
    """
    res = drone_base_orientation.inverse() * base_vector.to_Quaternionr() * drone_base_orientation
    return airsim.Vector3r(res.x_val, res.y_val, res.z_val)


def basis_oriented_quaternion(drone_base_orientation, base_quaternion):
    """
    performs a change of basis for airsim quaternions
    """
    return drone_base_orientation.inverse() * base_quaternion


def euler_to_quaternion(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    return airsim.Quaternionr(x_val=cy * cp * sr - sy * sp * cr,
                              y_val=sy * cp * sr + cy * sp * cr,
                              z_val=sy * cp * cr - cy * sp * sr,
                              w_val=cy * cp * cr + sy * sp * sr)


def quaternion_to_roll_pitch_without_yaw(q):
    # roll (x-axis rotation)
    sinr_cosp = 2 * (q.w_val * q.x_val + q.y_val * q.z_val)
    cosr_cosp = 1 - 2 * (q.x_val * q.x_val + q.y_val * q.y_val)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2 * (q.w_val * q.y_val - q.z_val * q.x_val)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    return roll, pitch


def quaternion_to_yaw(q):
    # yaw (z-axis rotation)
    siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw


def quaternion_to_euler(q):
    return quaternion_to_roll_pitch_without_yaw(q), quaternion_to_yaw(q)


def yaw_to_ned_quaternion(yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return airsim.Quaternionr(x_val=0.0, y_val=0.0, z_val=sy, w_val=cy)


def quat_to_ned_quat(q):
    return yaw_to_ned_quaternion(quaternion_to_yaw(q))


class airsimdroneracinglabEnv(Env):
    def __init__(self, config):
        """
        gym environment for the airsimdroneracinglab competition
        This version of the environment is not time-controlled and step() needs to be called repeatedly in a timer (real world setting, for the Real Time RL framework)
        The length of low_bound and high_bound is the number of actions
        Parameters are packed in the config dictionary
        Args (config dictionary keys):
            airsim_client: airsimdroneracinglab.MultiRotorClient: If None (needed for rllib compatibility), then a client is created by the environement
            control_method_str: a string that matches the name of the control API you want to use (e.g. 'moveByAngleRatesThrottleAsync'). (add support in _set_control_method_id() and _get_args_kwargs_control_method() and  when needed)
            low_bounds: numpy.array[(numpy.float32),...]: low bounds of all actions
            high_bounds: numpy.array[(numpy.float32),...]: high bounds of all actions
            ep_max_length: int: the max length of each episodes in timesteps
            drones_names: list of strings: list of agents names
            (optional) tier: int (default 0): competition tier (for access to everything to e.g. build a dataset, use tier 0)
            (optional) dummy_for_ip_port_file_init: True|False(default): if True, the environment will only be created to initialize the ip_port file. Apart from this specific use, this must be False
            (optional) rllib_compatibility: True|False(default): if True, the environment will only be initialized at the first reset() call
            (optional) rendering_mode: 'WINDOWED_DISPLAY'(default), 'WINDOWED_NO_DISPLAY', 'NO_DISPLAY' or 'NO_RENDER'
            (optional) time_step_method: 'JOIN'(default), 'CONTINUE_FOR_TIME'
            (optional) locker: True|False: whether a locker server is running and should be used (warning: if False, be sure to isolate workers or the program will attempt dangerous file access)
            (optional) act_in_obs: bool (default: True): whether the action should be appended to the observation
            (optional) default_act: action (default: None): action to append to obs at reset when act_in_obs is True
            (optional) act_preprocessor: function (default: None): preprocessor for individual actions before they are actually applied by step()
            (optional) obs_preprocessor: function (default: None): preprocessor for individual observations before they are returned by step()
            (optional) synchronous_actions: bool (default: True): whether time should be paused to apply actions simultaneously
            (optional) synchronous_states: bool (default: True): whether time should be paused to retrieve observations simultaneously
            (optional) obs_coord_system: string (default: 'dc'): coordinate system of the observations: 'dc' (fully drone-centric), 'ned' (no pitch/roll), 'global', 'all' (all coordinate systems)
            (optional) act_coord_system: string (default: 'dc'): coordinate system of the actions: 'dc' (fully drone-centric), 'ned' (no pitch/roll), 'global'. (add support in _get_args_kwargs_control_method() when needed)
            (optional) rf_config: dict (default: DEFAULT_RF_CONFIG): parameters dictionary of the reward function
            (optional) time_stop: bool (default: True): whether time should be stopped between steps
            (optional) real_time: bool (default: False): whether the action are for next time-step instead of current time-step
            (optional) act_threading: bool (default: True): whether actions are executed asynchronously in the RTRL setting. Set this to True when __apply_action_n() is a I/O operation blocking for the duration of an external time step
                Typically this is useful for the real world and for external simulators
                When this is True, __apply_action_n() should be a cpu-light I/O operation or python multithreading will slow down the calling program
                For cpu-intensive tasks (e.g. embedded simulators), this should be True only if you ensure that the CPU-intensive part is executed in another process while __apply_action_n() is only used for interprocess communications
            (optional) default_z_target: float (default: 0.0): initial Z target to make Z stabilizing APIs drone/ned-centric
        action_space is a gym.spaces.Tuple(gym.spaces.Box()) of length nb_drones
        observation_space is a gym.spaces.Tuple(gym.spaces.Dict()) of length nb_drones
        """
        if "dummy_for_ip_port_file_init" in config and config["dummy_for_ip_port_file_init"] is True:
            initialize_ip_port_file()
            print_with_pid("DEBUG: ip_file initialized")
            return
        print_with_pid("DEBUG: Creating new environment...")

        # what is initialized here is what is needed for rllib dummy environments

        self.config = config
        self.use_locker = config["use_locker"] if "use_locker" in config else True
        self.lock_client = LockerClient() if self.use_locker else None
        self.img_width = config["img_width"] if "img_width" in config else DEFAULT_IMG_WIDTH
        self.img_height = config["img_height"] if "img_height" in config else DEFAULT_IMG_HEIGHT
        self.time_stop = config["time_stop"] if "time_stop" in config else True
        self.real_time = config["real_time"] if "real_time" in config else False
        self.act_threading = config["act_thread"] if "act_thread" in config else True
        if not self.real_time:
            self.act_threading = False
        if self.act_threading:
            self._at_thread = Thread(target=None, args=(), kwargs={}, daemon=True)
            self._at_thread.start()  # dummy start for later call to join()
        self.act_in_obs = config["act_in_obs"] if "act_in_obs" in config else True
        self.default_act = config["default_act"] if "default_act" in config else None
        self.act_preprocessor = config["act_preprocessor"] if "act_preprocessor" in config else None
        self.obs_preprocessor = config["obs_preprocessor"] if "obs_preprocessor" in config else None
        self.synchronous_actions = config["synchronous_actions"] if "synchronous_actions" in config else True
        self.synchronous_states = config["synchronous_states"] if "synchronous_states" in config else True
        self.rf_config = config["rf_config"] if "rf_config" in config else DEFAULT_RF_CONFIG
        self._set_obs_coord_id(config["obs_coord_system"] if "obs_coord_system" in config else 'dc')
        self._set_act_coord_id(config["act_coord_system"] if "act_coord_system" in config else 'dc')
        self.process = None
        self.drones_names = config["drones_names"]
        self.nb_drones = len(self.drones_names)
        self.default_z_target = config["default_z_target"] if "default_z_target" in config else 0.0
        self.z_targets = [self.default_z_target, ] * self.nb_drones
        self.low_bounds = config["low_bounds"]
        self.high_bounds = config["high_bounds"]
        self.history_length = config["history_length"] if "history_length" in config else 3
        self.tier = config["tier"] if "tier" in config else 0
        assert 1 <= self.nb_drones <= 2, "Must have 1 or 2 drones"
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.initialized = False

        # Now in rllib compatibility mode we don't instantiate anything else here

        if "rllib_compatibility" not in config or not config["rllib_compatibility"]:
            print_with_pid("DEBUG: No rllib compatibility, initializing envrionement completely...")
            self._initialize()
        print_with_pid("DEBUG: New environment created")

    def _join_act_thread(self):
        """
        This is called at the beginning of every user-side API functions (step(), reset()...) for thread safety
        In the RTRL setting with action threading, this ensures that the previous time-step is completed when starting a new one
        """
        if self.act_threading:
            self._at_thread.join()

    def _apply_action_n(self, *args, **kwargs):
        """
        This is what must be called in step() to apply an action
        Call this with the args and kwargs expected by self.__apply_action_n()
        This in turn calls self.__apply_action_n()
        In RTRL action-threading, self.__apply_action_n() is called in a new Thread
        """
        if not self.act_threading:
            self.__apply_action_n(*args, **kwargs)
        else:
            self._at_thread = Thread(target=self.__apply_action_n, args=args, kwargs=kwargs)
            self._at_thread.start()

    def _set_control_method_id(self):
        """
        Each ID corresponds to a pattern in the signature of the API control function
        If you wish to add support for a new signature pattern, define a new ID here and modify _get_args_kwargs_control_method() accordingly
        """
        if self.control_method_str == 'moveByVelocityAsync':
            self.control_method_id = 1
        elif self.control_method_str == 'moveByRollPitchYawrateZAsync':
            self.control_method_id = 2
        else:
            # default pattern: actions are directly passed to args
            self.control_method_id = 0

    def _set_obs_coord_id(self, str):
        if str == 'all':
            self.obs_coord_id = 0  # all coordinate systems
        elif str == 'dc':
            self.obs_coord_id = 1  # drone-centric coordinates
        elif str == 'ned':
            self.obs_coord_id = 2  # ned coordinates
        else:
            self.obs_coord_id = 3  # global coordinates

    def _set_act_coord_id(self, str):
        if str == 'dc':
            self.act_coord_id = 1  # drone-centric coordinates
        elif str == 'ned':
            self.act_coord_id = 2  # ned coordinates
        else:
            self.act_coord_id = 3  # global coordinates

    def _initialize(self):
        """
        This is for rllib compatibility
        rllib will always create a dummy environment at the beginning just to get the action and observation spaces
        Of course we don't want a simulator to be instanciated for this dummy environment, so we create it only on the first call of reset()
        """
        self.initialized = True
        config = self.config
        if "time_step_method" in config:
            if config["time_step_method"] == "CONTINUE_FOR_TIME":
                self.time_step_method_id = 1
            else:
                self.time_step_method_id = 0
        else:
            self.time_step_method_id = 0
        self.histories = []
        for _ in range(self.nb_drones):
            self.histories.append(deque(maxlen=self.history_length))
        if "rendering_mode" in config:
            self.rendering_mode = config["rendering_mode"]
        else:
            self.rendering_mode = DEFAULT_RENDERING_MODE
        self.clock_speed = config["clock_speed"]
        self.airsim_client = config["airsim_client"]
        if not self.airsim_client:
            self.airsim_client, self.process = new_client(self.clock_speed, self.img_height, self.img_width, mode=self.rendering_mode, use_locker=self.use_locker, lock_client=self.lock_client)
        self.control_method_str = config["control_method_str"]
        self._set_control_method_id()
        self.control_method = getattr(self.airsim_client, self.control_method_str)
        self.ep_max_length = config["ep_max_length"]
        self.simulated_time_step = config["simulated_time_step"]
        self.cpu_time_step = self.simulated_time_step / self.clock_speed
        if "level_name" in config:
            self.level_name = config["level_name"]
        else:
            if np.random.randint(2) == 0:
                self.level_name = 'Soccer_Field_Easy'
            else:
                self.level_name = 'Soccer_Field_Medium'
        self.drones_offsets = config["drones_offsets"]
        self.current_objectives = None  # this is required for reset>RewardFunction to retrieve gate poses only once (because the API for this is prone to bugs)
        self.airsim_client.simLoadLevel(self.level_name)
        print_with_pid("DEBUG: confirmConnection()...")
        self.airsim_client.confirmConnection()  # failsafe
        time.sleep(2)  # let the environment load completely
        obs = self.reset()
        self.initial_objectives, self.gates_names = self.reward_functions[0].get_objectives()  # this is to save the initial configuration of the track
        self.current_objectives = copy.deepcopy(self.initial_objectives)
        print_with_pid("DEBUG: environment initialized")
        return obs

    def _kill_simulator(self):  # TODO: this doesn't work because msgpckrpc/tornado is not properly reset
        print_with_pid("DEBUG: Killing simulator processes")
        if SYS_STR == 'Linux':
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        else:
            os.kill(self.process.pid, 9)

    def _restart_simulator(self):  # TODO: this doesn't work. See _kill_simulator()
        """
        kills the simulator and starts a new instance
        """
        assert self.process is not None, "ERROR: the simulator process has not been started from the environment"
        self._kill_simulator()
        self.airsim_client, self.process = new_client(self.clock_speed, self.img_height, self.img_width, mode=self.rendering_mode, use_locker=self.use_locker, lock_client=self.lock_client)
        self.airsim_client.simLoadLevel(self.level_name)
        print_with_pid("DEBUG: confirmConnection()...")
        self.airsim_client.confirmConnection()  # failsafe
        time.sleep(2)  # let the environment load completely
        return self.reset()

    def _get_action_space(self):
        # print_with_pid("DEBUG: Getting action space")
        elt = spaces.Box(self.low_bounds, self.high_bounds)
        tup = (elt,) * self.nb_drones
        return spaces.Tuple(tup)

    def _get_observation_space(self):
        # print_with_pid("DEBUG: Getting observation space")
        elt = {}
        if self.tier <= 1:  # ground truth of everything
            if self.obs_coord_id <= 1:  # drone-centric
                elt['linear_velocity_dc'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['angular_velocity_dc'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['linear_acceleration_dc'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['angular_acceleration_dc'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['rival_position_dc'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['rival_orientation_dc'] = spaces.Box(low=-2.0, high=2.0, shape=(4,))
                elt['rival_linear_velocity_dc'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['rival_angular_velocity_dc'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['rival_linear_acceleration_dc'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['rival_angular_acceleration_dc'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['target_position_dc'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['target_orientation_dc'] = spaces.Box(low=-2.0, high=2.0, shape=(4,))
                elt['next_target_position_dc'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['next_target_orientation_dc'] = spaces.Box(low=-2.0, high=2.0, shape=(4,))
            if self.obs_coord_id == 0 or self.obs_coord_id == 2:  # NED
                elt['linear_velocity_ned'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['angular_velocity_ned'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['linear_acceleration_ned'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['angular_acceleration_ned'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['rival_position_ned'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['rival_orientation_ned'] = spaces.Box(low=-2.0, high=2.0, shape=(4,))
                elt['rival_linear_velocity_ned'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['rival_angular_velocity_ned'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['rival_linear_acceleration_ned'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['rival_angular_acceleration_ned'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['target_position_ned'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['target_orientation_ned'] = spaces.Box(low=-2.0, high=2.0, shape=(4,))
                elt['next_target_position_ned'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['next_target_orientation_ned'] = spaces.Box(low=-2.0, high=2.0, shape=(4,))
            if self.obs_coord_id == 0 or self.obs_coord_id == 3:  # global
                elt['linear_velocity_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['angular_velocity_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['linear_acceleration_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['angular_acceleration_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['rival_position_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['rival_orientation_glo'] = spaces.Box(low=-2.0, high=2.0, shape=(4,))
                elt['rival_linear_velocity_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['rival_angular_velocity_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['rival_linear_acceleration_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['rival_angular_acceleration_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['target_position_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['target_orientation_glo'] = spaces.Box(low=-2.0, high=2.0, shape=(4,))
                elt['next_target_position_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['next_target_orientation_glo'] = spaces.Box(low=-2.0, high=2.0, shape=(4,))
            elt['target_dims_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
            elt['next_target_dims_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
            if self.tier == 0:
                elt['front_camera_dc'] = spaces.Box(low=0.0, high=255.0, shape=(self.history_length, self.img_height, self.img_width, 3))
        else:  # tiers 2 and 3
            if self.obs_coord_id <= 1:  # drone-centric
                elt['linear_velocity_dc'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['angular_velocity_dc'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['linear_acceleration_dc'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['angular_acceleration_dc'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
            if self.obs_coord_id == 0 or self.obs_coord_id == 2:  # NED
                elt['linear_velocity_ned'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['angular_velocity_ned'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['linear_acceleration_ned'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['angular_acceleration_ned'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
            if self.obs_coord_id == 0 or self.obs_coord_id == 3:  # global
                elt['linear_velocity_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['angular_velocity_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['linear_acceleration_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
                elt['angular_acceleration_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
            elt['front_camera_dc'] = spaces.Box(low=0.0, high=255.0, shape=(self.history_length, self.img_height, self.img_width, 3))
        if self.obs_coord_id <= 2:  # in drone-centric observations, the gravity vector (ie orientation of the drone) is needed
            elt['gravity_angles_dc'] = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        if self.obs_coord_id == 0 or self.obs_coord_id == 3:  # in global observations, the position and orientation are needed
            elt['position_glo'] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
            elt['orientation_glo'] = spaces.Box(low=-2.0, high=2.0, shape=(4,))
        if self.act_in_obs:
            elt['action'] = self._get_action_space()[0]
        selt = spaces.Dict(elt)
        tup = (selt,) * self.nb_drones
        return spaces.Tuple(tup)

    def _get_imgs(self, camera_name, drone_idx):
        """
        gets the current image from camera_name, drone_name
        appends it to the history (builds it the first time)
        returns a copy of the history in a numpy array
        """
        cpt = 0
        try_again = True
        while try_again and cpt <= MAX_GETIMAGES_TRIALS:
            request = [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)]
            response = self.airsim_client.simGetImages(request, vehicle_name=self.drones_names[drone_idx])
            rec_img_height = response[0].height
            rec_img_width = response[0].width
            if rec_img_height >= 1 and rec_img_width >= 1:
                try_again = False
            else:
                cpt += 1
                print_with_pid("DEBUG: simGetImages failed, retrying...")

        img_rgb_1d1 = np.fromstring(response[0].image_data_uint8, dtype=np.uint8)
        img = img_rgb_1d1.reshape(rec_img_height, rec_img_width, 3)
        if len(self.histories[drone_idx]) != 0:
            self.histories[drone_idx].pop()
            self.histories[drone_idx].appendleft(img)
        else:
            for _ in range(self.history_length):
                self.histories[drone_idx].appendleft(img)
        return np.asarray(self.histories[drone_idx])

    def _get_obs(self, drone_idx, action):  # TODO: tiers 2 and 3
        """
        Returns the observation of drone drone_idx in the tier we are considering
        """
        # print_with_pid("DEBUG: Getting obs, drone_idx=", drone_idx)

        rf = self.reward_functions[drone_idx]
        curr_idx = rf.current_objective_idx
        if curr_idx >= len(rf.objectives):  # track complete
            curr_idx = len(rf.objectives) - 1
        next_idx = curr_idx + 1
        if next_idx >= len(rf.objectives):
            next_idx = curr_idx
        if self.tier != 2:
            opp_rf = self.reward_functions[1 - drone_idx]
        drone_base_position = rf.current_position
        drone_base_orientation = rf.current_kinematics.orientation
        elt = {}
        if self.tier <= 1:  # tiers 0 and 1
            if self.obs_coord_id <= 1:  # drone-centric
                elt['linear_velocity_dc'] = basis_oriented_vector(drone_base_orientation, rf.current_kinematics.linear_velocity).to_numpy_array()
                elt['angular_velocity_dc'] = basis_oriented_vector(drone_base_orientation, rf.current_kinematics.angular_velocity).to_numpy_array()
                elt['linear_acceleration_dc'] = basis_oriented_vector(drone_base_orientation, rf.current_kinematics.linear_acceleration).to_numpy_array()
                elt['angular_acceleration_dc'] = basis_oriented_vector(drone_base_orientation, rf.current_kinematics.angular_acceleration).to_numpy_array()
                elt['rival_position_dc'] = basis_oriented_vector(drone_base_orientation, opp_rf.current_position - drone_base_position).to_numpy_array()
                elt['rival_orientation_dc'] = basis_oriented_quaternion(drone_base_orientation, opp_rf.current_kinematics.orientation).to_numpy_array()
                elt['rival_linear_velocity_dc'] = basis_oriented_vector(drone_base_orientation, opp_rf.current_kinematics.linear_velocity).to_numpy_array()
                elt['rival_angular_velocity_dc'] = basis_oriented_vector(drone_base_orientation, opp_rf.current_kinematics.angular_velocity).to_numpy_array()
                elt['rival_linear_acceleration_dc'] = basis_oriented_vector(drone_base_orientation, opp_rf.current_kinematics.linear_acceleration).to_numpy_array()
                elt['rival_angular_acceleration_dc'] = basis_oriented_vector(drone_base_orientation, opp_rf.current_kinematics.angular_acceleration).to_numpy_array()
                elt['target_position_dc'] = basis_oriented_vector(drone_base_orientation, rf.objectives[curr_idx][0].position - drone_base_position).to_numpy_array()
                elt['target_orientation_dc'] = basis_oriented_quaternion(drone_base_orientation, rf.objectives[curr_idx][0].orientation).to_numpy_array()
                elt['next_target_position_dc'] = basis_oriented_vector(drone_base_orientation, rf.objectives[next_idx][0].position - drone_base_position).to_numpy_array()
                elt['next_target_orientation_dc'] = basis_oriented_quaternion(drone_base_orientation, rf.objectives[next_idx][0].orientation).to_numpy_array()
            if self.obs_coord_id == 0 or self.obs_coord_id == 2:  # NED
                drone_ned_orientation = quat_to_ned_quat(drone_base_orientation)
                elt['linear_velocity_ned'] = basis_oriented_vector(drone_ned_orientation, rf.current_kinematics.linear_velocity).to_numpy_array()
                elt['angular_velocity_ned'] = basis_oriented_vector(drone_ned_orientation, rf.current_kinematics.angular_velocity).to_numpy_array()
                elt['linear_acceleration_ned'] = basis_oriented_vector(drone_ned_orientation, rf.current_kinematics.linear_acceleration).to_numpy_array()
                elt['angular_acceleration_ned'] = basis_oriented_vector(drone_ned_orientation, rf.current_kinematics.angular_acceleration).to_numpy_array()
                elt['rival_position_ned'] = basis_oriented_vector(drone_ned_orientation, opp_rf.current_position - drone_base_position).to_numpy_array()
                elt['rival_orientation_ned'] = basis_oriented_quaternion(drone_ned_orientation, opp_rf.current_kinematics.orientation).to_numpy_array()
                elt['rival_linear_velocity_ned'] = basis_oriented_vector(drone_ned_orientation, opp_rf.current_kinematics.linear_velocity).to_numpy_array()
                elt['rival_angular_velocity_ned'] = basis_oriented_vector(drone_ned_orientation, opp_rf.current_kinematics.angular_velocity).to_numpy_array()
                elt['rival_linear_acceleration_ned'] = basis_oriented_vector(drone_ned_orientation, opp_rf.current_kinematics.linear_acceleration).to_numpy_array()
                elt['rival_angular_acceleration_ned'] = basis_oriented_vector(drone_ned_orientation, opp_rf.current_kinematics.angular_acceleration).to_numpy_array()
                elt['target_position_ned'] = basis_oriented_vector(drone_ned_orientation, rf.objectives[curr_idx][0].position - drone_base_position).to_numpy_array()
                elt['target_orientation_ned'] = basis_oriented_quaternion(drone_ned_orientation, rf.objectives[curr_idx][0].orientation).to_numpy_array()
                elt['next_target_position_ned'] = basis_oriented_vector(drone_ned_orientation, rf.objectives[next_idx][0].position - drone_base_position).to_numpy_array()
                elt['next_target_orientation_ned'] = basis_oriented_quaternion(drone_ned_orientation, rf.objectives[next_idx][0].orientation).to_numpy_array()
            if self.obs_coord_id == 0 or self.obs_coord_id == 3:  # global
                elt['linear_velocity_glo'] = rf.current_kinematics.linear_velocity.to_numpy_array()
                elt['angular_velocity_glo'] = rf.current_kinematics.angular_velocity.to_numpy_array()
                elt['linear_acceleration_glo'] = rf.current_kinematics.linear_acceleration.to_numpy_array()
                elt['angular_acceleration_glo'] = rf.current_kinematics.angular_acceleration.to_numpy_array()
                elt['rival_position_glo'] = opp_rf.current_position.to_numpy_array()
                elt['rival_orientation_glo'] = opp_rf.current_kinematics.orientation.to_numpy_array()
                elt['rival_linear_velocity_glo'] = opp_rf.current_kinematics.linear_velocity.to_numpy_array()
                elt['rival_angular_velocity_glo'] = opp_rf.current_kinematics.angular_velocity.to_numpy_array()
                elt['rival_linear_acceleration_glo'] = opp_rf.current_kinematics.linear_acceleration.to_numpy_array()
                elt['rival_angular_acceleration_glo'] = opp_rf.current_kinematics.angular_acceleration.to_numpy_array()
                elt['target_position_glo'] = rf.objectives[curr_idx][0].position.to_numpy_array()
                elt['target_orientation_glo'] = rf.objectives[curr_idx][0].orientation.to_numpy_array()
                elt['next_target_position_glo'] = rf.objectives[next_idx][0].position.to_numpy_array()
                elt['next_target_orientation_glo'] = rf.objectives[next_idx][0].orientation.to_numpy_array()
            elt['target_dims_glo'] = rf.objectives[curr_idx][1].to_numpy_array()
            elt['next_target_dims_glo'] = rf.objectives[next_idx][1].to_numpy_array()
            if self.tier == 0:  # additional ground truthes that are not in tier 1 for dataset collection
                elt['front_camera_dc'] = self._get_imgs(f"fpv_cam_{drone_idx + 1}", drone_idx)
        else:  # tiers 2 and 3
            if self.obs_coord_id <= 1:  # drone-centric
                elt['linear_velocity_dc'] = basis_oriented_vector(drone_base_orientation, rf.current_kinematics.linear_velocity).to_numpy_array()
                elt['angular_velocity_dc'] = basis_oriented_vector(drone_base_orientation, rf.current_kinematics.angular_velocity).to_numpy_array()
                elt['linear_acceleration_dc'] = basis_oriented_vector(drone_base_orientation, rf.current_kinematics.linear_acceleration).to_numpy_array()
                elt['angular_acceleration_dc'] = basis_oriented_vector(drone_base_orientation, rf.current_kinematics.angular_acceleration).to_numpy_array()
            if self.obs_coord_id == 0 or self.obs_coord_id == 2:  # NED
                drone_ned_orientation = quat_to_ned_quat(drone_base_orientation)
                elt['linear_velocity_ned'] = basis_oriented_vector(drone_ned_orientation, rf.current_kinematics.linear_velocity).to_numpy_array()
                elt['angular_velocity_ned'] = basis_oriented_vector(drone_ned_orientation, rf.current_kinematics.angular_velocity).to_numpy_array()
                elt['linear_acceleration_ned'] = basis_oriented_vector(drone_ned_orientation, rf.current_kinematics.linear_acceleration).to_numpy_array()
                elt['angular_acceleration_ned'] = basis_oriented_vector(drone_ned_orientation, rf.current_kinematics.angular_acceleration).to_numpy_array()
            if self.obs_coord_id == 0 or self.obs_coord_id == 3:  # global
                elt['linear_velocity_glo'] = rf.current_kinematics.linear_velocity.to_numpy_array()
                elt['angular_velocity_glo'] = rf.current_kinematics.angular_velocity.to_numpy_array()
                elt['linear_acceleration_glo'] = rf.current_kinematics.linear_acceleration.to_numpy_array()
                elt['angular_acceleration_glo'] = rf.current_kinematics.angular_acceleration.to_numpy_array()
            elt['front_camera_dc'] = self._get_imgs(f"fpv_cam_{drone_idx + 1}", drone_idx)
        if self.obs_coord_id <= 2:  # in drone-centric and ned observations, the gravity vector is needed (orientation of the drone)
            elt['gravity_angles_dc'] = np.array(quaternion_to_roll_pitch_without_yaw(drone_base_orientation))
        if self.obs_coord_id == 0 or self.obs_coord_id == 3:  # in global observations, the position and orientation are needed
            elt['position_glo'] = drone_base_position.to_numpy_array()
            elt['orientation_glo'] = drone_base_orientation.to_numpy_array()
        if self.act_in_obs:
            elt['action'] = action

        if self.obs_preprocessor is not None:
            elt = self.obs_preprocessor(elt)
        return elt

    def _update_states_and_get_rewards_and_dones(self):
        # print_with_pid("DEBUG: update state and get rew and dones")
        for reward_function in self.reward_functions:
            reward_function.update_state()
        rew_n = []
        done_n = []
        for reward_function in self.reward_functions:
            rew_n.append(reward_function.get_reward())  # can change parameters here
            done_n.append(reward_function.done)
        return rew_n, done_n

    def _gates_randomization(self):
        """
        This randomizes gates during reset()
        Also contains a workaround that makes the last gate unreachable to avoid airsim bugs/crashes on their reset function
        """
        if self.current_objectives is None:
            pass
        elif self.level_name == 'Soccer_Field_Medium':
            self.current_objectives = copy.deepcopy(self.initial_objectives)

            # delete last gate:
            # gate_pose = self.current_objectives[-1][0]
            # gate_pose.position.x_val = 1000.0
            # gate_pose.position.y_val = 1000.0
            # gate_pose.position.z_val = 1000.0
            # self.airsim_client.simSetObjectPose(self.gates_names[-1], gate_pose)
            # self.current_objectives.pop()

            # randomize first gate (old)
            # randomize all gates
            nb_gates = len(self.current_objectives)
            gate_pose = self.current_objectives[0][0]
            x_noise = (np.random.random() - 0.5) * 10.0  # meters
            y_noise = (np.random.random() - 0.5) * 10.0 + 5.0  # meters
            z_noise = (np.random.random() - 0.5) * -2.0  # meters
            gate_pose.position.x_val += x_noise
            gate_pose.position.y_val += y_noise
            gate_pose.position.z_val = min(gate_pose.position.z_val + z_noise, 0.0)  #check out here

            x_euler_angle_noise = 0.0  # radian
            y_euler_angle_noise = 0.0  # radian
            z_euler_angle_noise = (np.random.random() - 0.5) * 0.5 * np.pi  # radian
            quaternion_noise = euler_to_quaternion(x_euler_angle_noise, y_euler_angle_noise, z_euler_angle_noise)
            gate_pose.orientation = quaternion_noise  # FIXME: the initial pose of the first gate is by default an invalid orientation (w=1.0 and axis=0.0) that cannot be rotated

            self.airsim_client.simSetObjectPose(self.gates_names[0], gate_pose)
                
            for i in range(nb_gates-1):
                
                # randomize second gate:
                previous_gate_pose = copy.deepcopy(gate_pose)
                self.current_objectives[i+1][0] = previous_gate_pose
                gate_pose = self.current_objectives[i+1][0]
    
                x_noise = (np.random.random() - 0.5) * 10.0  # meters
                y_noise = (np.random.random() - 0.5) * 10.0 + 10.0  # meters
                z_noise = (np.random.random() - 0.5) * -2.0  # meters
                vector_noise = basis_oriented_vector(gate_pose.orientation.inverse(), airsim.Vector3r(x_val=x_noise, y_val=y_noise, z_val=0.0))
                gate_pose.position = gate_pose.position + vector_noise
                gate_pose.position.z_val = z_noise
    
                x_euler_angle_noise = 0.0  # radian
                y_euler_angle_noise = 0.0  # radian
                z_euler_angle_noise = (np.random.random() - 0.5) * 0.5 * np.pi  # radian
                quaternion_noise = euler_to_quaternion(x_euler_angle_noise, y_euler_angle_noise, z_euler_angle_noise) * gate_pose.orientation
                gate_pose.orientation = quaternion_noise

                self.airsim_client.simSetObjectPose(self.gates_names[i+1], gate_pose)

    def _get_args_kwargs_control_method(self, action_n, idx):
        """
        This is where actions are interpreted in order to be passed to the control control_method
        For example, we convert local actions to global actions for global API functions here
        control_method_id must be set in _set_control_method_id()
        also applies action proprocessor if any
        """
        kwargs = {'duration': self.simulated_time_step,
                  'vehicle_name': self.drones_names[idx]}
        act = action_n[idx]
        if self.act_preprocessor is not None:
            act = self.act_preprocessor(act)
        if self.control_method_id == 0:  # default behavior
            args = tuple(act)
        elif self.control_method_id == 1:  # moveByVelocityAsync behavior
            rf = self.reward_functions[idx]
            drone_base_orientation = rf.current_kinematics.orientation
            yaw_rate = act[3]
            if self.act_coord_id == 1:  # drone-centric coordinates
                act_o = basis_oriented_vector(drone_base_orientation.inverse(), airsim.Vector3r(act[0], act[1], act[2]))  # TODO: check that this is correct
            elif self.act_coord_id == 2:  # NED
                drone_ned_orientation = quat_to_ned_quat(drone_base_orientation)
                act_o = basis_oriented_vector(drone_ned_orientation.inverse(), airsim.Vector3r(act[0], act[1], act[2]))  # TODO: check that this is correct
            else:  # global coordinates
                act_o = airsim.Vector3r(act[0], act[1], act[2])
            args = tuple([act_o.x_val, act_o.y_val, act_o.z_val])
            kwargs['yaw_mode'] = {'is_rate': True, 'yaw_or_rate': yaw_rate}
        elif self.control_method_id == 2:  # moveByRollPitchYawrateZAsync behavior
            rf = self.reward_functions[idx]
            act_f = copy.deepcopy(act)  # ned and global coordinates
            if self.act_coord_id == 1 or self.act_coord_id == 2:  # drone-centric or NED coordinates
                self.z_targets[idx] = self.z_targets[idx] + act[3]
                act_f[3] = self.z_targets[idx]
            args = tuple(act_f)
        return args, kwargs

    def _clip_action_n(self, action_n):
        for i, action in enumerate(action_n):
            if not self.action_space[i].contains(action):
                print(f"DEBUG: action_n:{action_n} not in action space:{self.action_space} clipping...")
                for j, low_bound in enumerate(self.low_bounds):
                    diff_low = action[j] - low_bound
                    if diff_low < 0:
                        action_n[i][j] = low_bound
                for j, high_bound in enumerate(self.high_bounds):
                    diff_high = high_bound - action[j]
                    if diff_high < 0:
                        action_n[i][j] = high_bound
        return action_n

    def __apply_action_n(self, action_n, idxs):
        """
        This function applies the control API to all the drones with parameters action_n
        idxs is a [] of randomly sorted drone indices
        action bounds need to be chosen according to the chosen API control when instantiating the gym environment
        !: this function is the target of a Thread just before step() returns in the RTRL setting when self.act_thread is True
            In this specific case, all subsequent calls to the environment will join this Thread for general thread-safety
        """
        # print_with_pid('DEBUG: apply action_n')
        self.airsim_client.simPause(True) if self.synchronous_actions else self.airsim_client.simPause(False)
        action_n = self._clip_action_n(action_n)
        f_n = []
        for i in idxs:
            ff = None
            if not self.reward_functions[i].done:
                args, kwargs = self._get_args_kwargs_control_method(action_n, i)
                ff = self.control_method(*args, **kwargs)
            f_n.append(ff)
        if self.time_step_method_id == 1:  # continueForTime / sleep method # FIXME: the actual time-step is hardware-dependent here because clockspeed is not really respected in the simulator, and simContinueForTime is bug-prone
            if self.time_stop:
                if not self.synchronous_actions:
                    self.airsim_client.simPause(True)
                self.airsim_client.simContinueForTime(self.cpu_time_step)
            else:
                if self.synchronous_actions:
                    self.airsim_client.simPause(False)
                time.sleep(self.cpu_time_step)
        else:  # join method # FIXME: things happen in simulation during the time it takes to retrieve the join status and to send the simPause command
            if self.synchronous_actions:
                self.airsim_client.simPause(False)
            for f in f_n:
                if f is not None:
                    f.join()
            if self.time_stop:
                self.airsim_client.simPause(True)

    def _get_states(self, action_n):
        """
        updates the states and returns the transition outcome
        """
        # print_with_pid('DEBUG: get transition')
        self.airsim_client.simPause(True) if self.synchronous_states else self.airsim_client.simPause(False)
        rew_n, done_n = self._update_states_and_get_rewards_and_dones()
        obs_n = []
        info_n = []
        for drone_idx in range(len(self.drones_names)):
            obs_n.append(self._get_obs(drone_idx, action_n[drone_idx]))
            info_n.append({})  # we can put gym debug info here
        if self.current_step >= self.ep_max_length:
            for i in range(len(done_n)):
                done_n[i] = True
        if not self.death_occurred:
            for i in range(len(done_n)):
                if self.reward_functions[i].death:
                    # TODO: check that the dead vehicle has indeed been detected as dead in the software
                    self.airsim_client.disarm(vehicle_name=self.drones_names[i])
                    self.death_occurred = True
        self.airsim_client.simPause(True) if self.time_stop else self.airsim_client.simPause(False)
        return obs_n, rew_n, done_n, info_n

    def _step_common_begin(self):  # TODO : cap action whithin self.action_space
        """
        This function outputs a random order on drones (for calling the API in random order in real time control)
        """
        self.current_step += 1
        idxs = list(range(self.nb_drones))
        random.shuffle(idxs)
        return idxs

    # gym user API functions:

    def reset(self):  # TODO: add support for random initial drone placement and gate randomization
        """
        Use reset() to reset the environment
        !: compatible only with 1 or 2 drones
        """
        self._join_act_thread()
        # print_with_pid("DEBUG: called env reset()")
        if not self.initialized:
            # print_with_pid("DEBUG: rllib compatibility")
            return self._initialize()

        self.current_step = 0
        self.death_occurred = False
        self.z_targets = [self.default_z_target,] * self.nb_drones

        # caution: airsim_client.reset() seems to make Airsim crash quite often
        # see: https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/issues/60 :

        # signal.alarm(RESET_TIMEOUT + 1)  # enable alarm
        # print_with_pid("DEBUG: called simPause(False)")
        self.airsim_client.simPause(False)
        # print_with_pid("DEBUG: called sim reset()")
        self.airsim_client.reset()
        # print_with_pid("DEBUG: called simResetRace()")
        self.airsim_client.simResetRace()
        time.sleep(SLEEP_TIME_AT_RESETRACE)
        # print_with_pid("DEBUG: called simPause(True)")
        self.airsim_client.simPause(True)

        # print_with_pid("DEBUG: called _gates_randomization()")
        self._gates_randomization()

        for drone_name in self.drones_names:
            self.airsim_client.enableApiControl(vehicle_name=drone_name)
            self.airsim_client.arm(vehicle_name=drone_name)

        obs = []
        self.reward_functions = []  # must be initialized after first call to simResetRace() for tiers 2 and 3
        for drone_idx, drone_name in enumerate(self.drones_names):
            self.reward_functions.append(rf.RewardFunction(airsim_client=self.airsim_client, vehicle_name=drone_name, base_offset=self.drones_offsets[drone_idx], objectives=self.current_objectives, param_dict=self.rf_config))
        if len(self.reward_functions) == 2:  # in the multiagent setting, we have to set opponent reward functions
            self.reward_functions[0].set_opponent_RewardFunction(self.reward_functions[1])
            self.reward_functions[1].set_opponent_RewardFunction(self.reward_functions[0])
        for drone_idx in range(len(self.drones_names)):
            obs.append(self._get_obs(drone_idx, self.default_act))
        # print_with_pid(f'DEBUG: called simStartRace(tier={self.tier})')
        self.airsim_client.simStartRace(tier=1 if self.tier == 0 else self.tier, competitor=False)  # must be called the first time after instantiating reward functions or reward functions will be broken in tier 2 and 3 (because of noisy gate estimates)
        self.airsim_client.simPause(True) if self.time_stop else self.airsim_client.simPause(False)
        # print_with_pid('DEBUG: PauseEnd')
        # print_with_pid('DEBUG: end reset')
        return obs

    def step(self, action):
        """
        Call this function to perform a step
        :param action: numpy.array[n_drones][n_actions] values for each action of each drone
        returns: obs_n, rew_n, done_n, either of the CURRENT (if not real_time) or PREVIOUS (if real_time) transition: see real-time RL
        """
        self._join_act_thread()
        idxs = self._step_common_begin()
        if not self.real_time:
            self._apply_action_n(action, idxs)
        obs_n, rew_n, done_n, info_n = self._get_states(action)
        if self.real_time:
            self._apply_action_n(action, idxs)
        return obs_n, rew_n, done_n, info_n

    def stop(self):
        self._join_act_thread()
        if self.process is not None:
            print_with_pid(f"DEBUG: call to stop(). Calling _kill_simulator()")
            self._kill_simulator()
            self.airsim_client = None
            self.process = None

    def render(self, mode='human', camera_name='fpv_cam_1', drone_idx=0):  # TODO: render should visually show the current state of the environment (should not use simGetImages)
        self._join_act_thread()
        request = [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)]
        response = self.airsim_client.simGetImages(request, vehicle_name=self.drones_names[drone_idx])
        img_rgb_1d1 = np.fromstring(response[0].image_data_uint8, dtype=np.uint8)
        img = img_rgb_1d1.reshape(response[0].height, response[0].width, 3)
        print("image shape: ", img.shape)
