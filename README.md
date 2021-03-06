# Gym Game Of Drones

An advanced gym environment and a Batch RL dataset collector game made on top of the Airsim Drone Racing Lab binaries.

# Installation:

```bash
git clone https://github.com/yannbouteiller/gym-airsimdroneracinglab.git
cd gym-airsimdroneracinglab
pip install -e .
python download_and_mod_airsim.py
```
Note 0: `python download_and_mod_airsim.py` may need to be executed as administrator

Note 1: Execute these in this order because the second script will mod the airsimdroneracinglab installation previously performed by pip

Note 2: The AirSim competition binaries are not open source and will be downloaded by download_and_mod_airsim.py.
If you are just updating your installation and don't want to download them again, use instead:

```bash
python download_and_mod_airsim.py --no-download
```

# Gym Environment

The embedded Gym environment is highly customizable through the config dictionary. Read the [environment's code](https://github.com/yannbouteiller/gym-airsimdroneracinglab/blob/main/gym_game_of_drones/envs/multi_agent/gym_airsimdroneracinglab/airsimdroneracinglab_env.py) for documentation.

This environment has been conceived with the [RTRL framework](https://arxiv.org/abs/1911.04448) in mind, and is fully compatible with its setting.

# Game Of Datasets

The 'Game of Drones' project has given birth to a pretty advanced gym environment. The dataset collector script turns this gym environment into a multiplayer videogame that can be used to collect Offline Reinforcement Learning datasets in all imaginable settings, including real-time. 

## How to run the dataset collector game:

```bash
cd gym_game_of_drones
python dataset_collector_game.py [options]
```
Each time you launch the script, give a new experiment name with the --experiment_name option, or the collected data will erase previous data. For instance:
```
--experiment_name="building99_0"
--experiment_name="building99_1"
--experiment_name="building99_2"
--experiment_name="Soccer_Field_Medium_0" --level_name="Soccer_Field_Medium"
...
```
Please also record the players' names with the following options (player 1 is on the left of the screen, player 2 is on the right):
```
--player_1
--player_2
```

You can choose the map by using this option:

```
--level_name
```
There are 4 possibilities: "Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard", the defalut map is "Building99_Hard". 

**IMPORTANT**: if your computer is too slow at rendering images (ZhangJiaJie is particularily slow), AirSim will reset the race and trap you in a box after 300 seconds, which cannot be fixed for now. If this happens to you, please discard the data by pressing "Q" and use simpler maps that you can complete in less than 300 seconds.

What's more, we are happy if you are using the **"Soccer_Field_Medium"** and **"Building99_Hard"** maps, because we are planning to focus on these two.

In the end, your command should look something like, for example:

```
python dataset_collector_game.py --level_name "Soccer_Field_Medium" --experiment_name "Soccer_Field_Medium_0" --player_1 "joyfly" --player_2 "superhero"
```

If you have two joysticks, connect them before launching the game. You can press the "C" key when the game is running to configure your joysticks.

## Important keys:
```
C: configure joysticks
Q: reset the game / discard the current episode
ESC: exit the program
S: toggle dataset reccording
``` 
After you are familiar with the map, you can press the "S" key to start recording the dataset.
Reseting with "Q" will discard the current episode if you are recording.
Enjoy playing the game!


The keyboard controls are:


```python
# general controls:

EXIT = pg.K_ESCAPE # quit game
RESET = pg.K_q  # resets (and discards) episode
SAVE_EPISODES = pg.K_s  # toggles dataset recording
CONFIG_JS = pg.K_c  # calls joystick configuration

# keyboard controls player 1:

FORWARD1 = pg.K_y
BACKWARD1 = pg.K_h
RIGHT1 = pg.K_u
LEFT1 = pg.K_t
RIGHTYAW1 = pg.K_j
LEFTYAW1 = pg.K_g
UP1 = pg.K_o
DOWN1 = pg.K_l

# keyboard controls player 2:

FORWARD2 = pg.K_KP8
BACKWARD2 = pg.K_KP5
RIGHT2 = pg.K_KP9
LEFT2 = pg.K_KP7
RIGHTYAW2 = pg.K_KP6
LEFTYAW2 = pg.K_KP4
UP2 = pg.K_PAGEUP
DOWN2 = pg.K_PAGEDOWN
```

## Dataset feedback: 

After you are done recording, please send the dataset folder to Dong or Yann. The path should be 

```
\YOUR_PATH_TO_THIS_FOLDER\gym_game_of_drones\gym_game_of_drones\drone_racing_dataset_collector\dataset\*.pkl
```

# Multi-Agent Gym environments

The environment is compatible with rllib, which handles multi-agent and requires a special interface that is not (yet) a default for multiagent gym environments, as gym has initially been designed for single agent environments. The "wrapper" (not an actual gym wrapper but that can be done easily) present in envs allows you to wrap the environment for rllib compatibilty.

A multi-agent training should in theory lead to inference and adaptation to unknown opponent policies. A whole lot of work from OpenAI and and Deepmind shows that.

Some interesting theoretical ressources:
- MADDPG ( https://arxiv.org/pdf/1706.02275.pdf )
- LOLA ( https://arxiv.org/pdf/1709.04326.pdf )
- LOLA-Dice ( https://arxiv.org/pdf/1802.05098.pdf )

# Parallel training with rllib:

The environment has additional features in order to support spawning multiple instances of AirSim for parralel training with rllib.

# Issues:

It is very likely that the quality of the simulation is hardware-dependent. See https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/issues/17

Random crashes may happen on resets. See https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/issues/60

The way the simulator handles inter-drone collision and disqualification may be weird compared to what is described in the rules. See https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/issues/43

AirSim is very slow at rendering camera images, which makes the dataset collector game very laggy (by default, it should run at 10Hz if images were rendered instantly). You can benchmark the gym environment using the provided benchmark script, or benchmark the game by setting the benchmark option.
