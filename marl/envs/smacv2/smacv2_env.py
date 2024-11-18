from __future__ import absolute_import, division, print_function

import sys
from os import replace

import numpy as np
from absl import logging


logging.set_verbosity(logging.DEBUG)
import os.path as osp
from pathlib import Path
import yaml

from gym.spaces import Box, Discrete


from .starcraft2 import StarCraft2Env
from .wrapper import StarCraftCapabilityEnvWrapper

class SMACv2Env:
    def __init__(self, args):
        self.map_config = self.load_map_config(args["map_name"])
        self.algorithm_name = args["algorithm_name"] 

    def step(self, actions):
        processed_actions = np.squeeze(actions, axis=1).tolist()
        reward, terminated, info = self.env.step(actions)
        
        if self.algorithm_name == "mast":
            obs = self.env.get_own_obs()
            state = None
            
            """
            obs_enemies: (self.n_enemies, self.n_agents)
            obs_ally: (self.n_agents, self.n_agents)
            """
        else:
            obs = self.env.get_obs()
            state = self.repeat(self.env.get_state())
            
        rewards = [[reward]] * self.n_agents

        info["bad_transition"] = False
        if terminated:
            if self.env.env.timeouts > self.timeouts:
                assert (
                    self.env.env.timeouts - self.timeouts == 1
                ), "Change of timeouts unexpected."
                info["bad_transition"] = True
                self.timeouts = self.env.env.timeouts

        infos = [info] * self.n_agents

        avail_actions = self.env.get_avail_actions()
        
        dones = []
        total_dones = []
        for i in range(self.env.n_agents):
            if terminated:
                dones.append(True)
                total_dones.append(True)
            else:
                dones.append(self.env.death_tracker_ally[i])
                total_dones.append(self.env.death_tracker_ally[i])
                
        for i in range(self.env.n_enemies):
            if self.env.death_tracker_enemy[i]:
                total_dones.append(True)
            else:
                total_dones.append(self.env.death_tracker_enemy[i])
                
        visible_masking = self.env.get_visible_object()
                
        return obs, state, rewards, dones, total_dones, infos, avail_actions, visible_masking

    def reset(self):
        self.env.reset()
        if self.algorithm_name == "mast":
            obs = self.env.get_own_obs()
            state = None
        else:
            obs = self.env.get_obs()
            state = self.repeat(self.env.get_state())
        avail_actions = self.env.get_avail_actions()
        
        visible_masking = self.env.get_visible_object()
        
        return obs, state, avail_actions, visible_masking

    def seed(self, seed):
        self.env = StarCraftCapabilityEnvWrapper(
            seed=seed, 
            algorithm_name = self.algorithm_name, 
            **self.map_config
        )
        env_info = self.env.get_env_info()
        n_actions = env_info["n_actions"]
        state_shape = env_info["state_shape"]
        obs_shape = env_info["obs_shape"]
        self.n_agents = env_info["n_agents"]
        self.timeouts = self.env.env.timeouts

        self.share_observation_space = self.repeat(
            Box(low=-np.inf, high=np.inf, shape=(state_shape,))
        )
        self.observation_space = self.repeat(
            Box(low=-np.inf, high=np.inf, shape=(obs_shape,))
        )
        self.action_space = self.repeat(Discrete(n_actions))
        
        self.objects = self.env.n_agents + self.env.n_enemies

    def close(self):
        self.env.close()

    def load_map_config(self, map_name):
        base_path = osp.split(osp.split(osp.dirname(osp.abspath(__file__)))[0])[0]
        map_config_path = (
            Path(base_path)
            / "configs"
            / "envs_cfgs"
            / "smacv2_map_config"
            / f"{map_name}.yaml"
        )
        with open(str(map_config_path), "r", encoding="utf-8") as file:
            map_config = yaml.load(file, Loader=yaml.FullLoader)
        return map_config

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]