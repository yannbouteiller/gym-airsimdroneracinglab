import gym
from ray.rllib.env import MultiAgentEnv
import numpy as np


class RllibMultiAgentEnv(MultiAgentEnv):
    """
    Multi-agent compatibility layer for RLLIB
    Also calls stop() (ie kill the simulator) when a TimeoutError is raised from msgpckrpc
    """
    def __init__(self, env_name, config):
        """
        :param env_name: (string) the name of the gym multiagent environement
        :param config: (dict) a dictionnary of config parameters for the gym environement
        config must contain a field 'agents_names': list of agents names in the right order
        """
        if "qualifier" in env_name:
            self.qualifier = True
        else:
            self.qualifier = False
        if "synchronous_dones" in config:
            self.synchronous_dones = config["synchronous_dones"]
        else:
            self.synchronous_dones = False
        try:
            self.env = gym.make(env_name, config=config)
        except Exception as e:
            print("rllib init() caught exception ", e)
            print("DEBUG: Error occurred in gym.make(), now calling stop()")
            self.stop()
            raise
        self.agents_names = config["agents_names"]
        self.tier = config["tier"] if "tier" in config else None
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range if hasattr(self.env, 'reward_range') else [-np.inf, np.inf]
        self.metadata = self.env.metadata if hasattr(self.env, 'metadata') else {'render.modes': ['human']}

    def stop(self):
        print("DEBUG: call to rllib multiagent compatibility layer stop()")
        self.env.stop()

    def reset(self):
        """Resets the env and returns observations from ready agents.
        Returns:
            obs (dict): New observations for each ready agent.
        """
        try:
            zippedObs = zip(self.agents_names, self.env.reset())
        except Exception as e:
            print("rllib wrapper reset() caught exception", e)
            print("DEBUG: Error occurred in reset(), now calling stop()")
            self.stop()
            raise
        return dict(zippedObs)

    def step(self, action_dict):
        """Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.
        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """
        action_n = []
        alive_agents_idxs = []
        alive_agents_names = []
        for idx, agent_name in enumerate(self.agents_names):
            if agent_name in action_dict:
                action = action_dict[agent_name]
                if type(action[0]).__module__ == 'numpy':  # we need to denumpyze for msgpck to work (?)
                    action_bis = []
                    for i, act in enumerate(action):
                        action_bis.append(act.item())
                    action = action_bis
                action_n.append(action)
                alive_agents_idxs.append(idx)
                alive_agents_names.append(agent_name)
            else:
                action_n.append(self.action_space[agent_name].sample())
        try:
            s_obs_n, s_rew_n, s_done_n, s_info_n = self.env.step(action_n)
        except Exception as e:
            print("rllib wrapper step() caught exception", e)
            print("DEBUG: Error occurred in reset(), now calling stop()")
            self.stop()
            raise

        for i in range(len(s_info_n)):
            s_info_n[i]["done"] = s_done_n[i]
        if self.synchronous_dones:
            if False in s_done_n:
                for i in range(len(s_done_n)):
                    s_done_n[i] = False
        if len(alive_agents_idxs) == len(self.agents_names):
            obs_n, rew_n, done_n, info_n = s_obs_n, s_rew_n, s_done_n, s_info_n
        else:
            obs_n, rew_n, done_n, info_n = [], [], [], []
            for idx in alive_agents_idxs:
                obs_n.append(s_obs_n[idx])
                rew_n.append(s_rew_n[idx])
                done_n.append(s_done_n[idx])
                info_n.append(s_info_n[idx])
        zippedObs = zip(alive_agents_names, obs_n)
        zippedRew = zip(alive_agents_names, rew_n)
        zippedDones = zip(alive_agents_names, done_n)
        zippedInfo = zip(alive_agents_names, info_n)
        dictDones = dict(zippedDones)
        if False not in done_n:
            dictDones["__all__"] = True
        else:
            dictDones["__all__"] = False

        return dict(zippedObs), dict(zippedRew), dictDones, dict(zippedInfo)

    def get_unwrapped(self):
        return [self.env]
