import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='gym-airsimdroneracinglab-v0',
    entry_point='gym_game_of_drones.envs.multi_agent.gym_airsimdroneracinglab:airsimdroneracinglabEnv',
)
