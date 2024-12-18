import os 
import re
import wandb
import torch
import socket
import setproctitle

import numpy as np
import torch.nn as nn
from gym import spaces

from utils.util import timer_start, timer_cancel

from utils.shared_buffer import SharedReplayBuffer

from envs import LOGGER_REGISTRY

from typing import Dict, Any
from utils.envs_tools import (
    set_seed, 
    make_train_env, 
    make_eval_env,  
    make_render_env, 
    get_num_agents
)
from utils.configs_tools import init_dir, save_config

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """Initialize Base Runner class.
    Args:
        args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
        algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
        env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
    """
    def __init__(self, exp_args, algo_args, env_args):
        
        timer = timer_start()
        # exp args
        self.exp_args: Dict[str, Any] = exp_args
        self.algo_args: Dict[str, Any] = algo_args
        self.env_args: Dict[str, Any] = env_args
        
        self.cuda: bool = self.exp_args["cuda"] 
        self.use_gpu: bool = self.exp_args["use_gpu"]   
        self.use_wandb: bool = self.exp_args["use_wandb"]
        self.use_render: bool = self.exp_args["use_render"]
        self.save_model: bool = self.exp_args["save_model"]
        
        self.seed: int = self.exp_args["seed"] 
        self.num_gpu: int = self.exp_args["num_gpu"]    
        self.log_interval: int = self.exp_args["log_interval"]  
        self.eval_interval: int = self.exp_args["eval_interval"]
        self.torch_threads: int = self.exp_args["torch_threads"]    
        self.render_episodes: int = self.exp_args["render_episodes"] 
        
        self.save_interval: float = self.exp_args["save_interval"]    
        
        self.log_dir: str = self.exp_args["log_dir"]
        self.map_name: str = self.exp_args["map_name"]
        self.env_name: str = self.exp_args["env_name"]
        self.model_dir: str = self.exp_args["model_dir"]
        self.user_name: str = self.exp_args["user_name"]
        self.group_name: str = self.exp_args["group_name"]
        self.algorithm_name: str = self.exp_args["algorithm_name"]
        self.experiment_name: str = self.exp_args["experiment_name"] 
        self.curriculum_envs: list = self.exp_args["curriculum_envs"]  
        
        # algo args
        self.use_eval: bool = self.algo_args["eval"]["use_eval"] 
        self.use_linear_lr_decay: bool = self.algo_args["train"]["use_linear_lr_decay"]
           
        self.episode_length: int = self.algo_args["train"]["episode_length"]    
        self.n_rollout_threads: int = self.algo_args["train"]["n_rollout_threads"]   
        self.n_eval_rollout_threads : int = self.algo_args["eval"]["n_eval_rollout_threads"]  
        self.recurrent_N: int = self.algo_args["model"]["recurrent_N"]
        self.hidden_size: int = self.algo_args["model"]["hidden_size"]  
        self.eval_episodes: int = self.algo_args["eval"]["eval_episodes"]
        self.module_names: str = self.algo_args["model"]["module"]
        # env args
        self.num_env_steps: int = self.env_args["num_env_steps"]
        

        set_seed(seed = self.seed)

        if not self.use_render:
            self.run_dir, self.save_dir = init_dir(
                env_name = self.env_name,
                map_name = self.curriculum_envs[-1],
                algo = self.algorithm_name,
                exp_name = self.experiment_name,
                seed = self.seed,
                logger_path=self.log_dir,
            )
            save_config(
                args = self.exp_args, 
                algo_args = self.algo_args, 
                env_args = self.env_args, 
                run_dir = self.run_dir
            )

        setproctitle.setproctitle(
            str(self.algorithm_name) + "-" + str(self.env_name) + "-" + str(self.experiment_name)
        )
        
        if self.use_wandb and not self.use_render:
            self.wandb = wandb.init(
                config={**self.exp_args, **self.algo_args, **self.env_args},
                project= self.env_name + "_" + self.curriculum_envs[-1],     
                notes=socket.gethostname(),
                entity=self.user_name,
                name="-".join([self.algorithm_name, self.experiment_name, "seed" + str(self.seed)]),
                group=self.group_name,
                dir=str(self.run_dir),
                job_type="training"
            )
            
        #------------------------------------envs settting------------------------------------#
        
        self.use_curriculum = False
        if self.exp_args["map_name"] == "curriculum_learning":  
            self.use_curriculum = True
            self.weighted_win_rate = 0.0
            self.decay_rate = 0.2
            self.threshold = 0.6
            self.curriculum_level = 0
            self.stack = 0
            self.stack_threshold = 5
            self.map_name = self.curriculum_envs[self.curriculum_level]
            self.exp_args["map_name"] = self.map_name
            
        # united args
        self.policy_args: Dict[str, Any] = {**self.exp_args, **self.algo_args["train"], **self.algo_args["model"], **self.algo_args["algo"]}
        self.args: Dict[str, Any] = {**self.exp_args, **self.algo_args, **self.env_args}
        self.env_info: Dict[str, Any] = {"map_name":self.map_name, "algorithm_name":self.algorithm_name}

        if self.use_gpu and torch.cuda.is_available():
            print("Using GPU...")
            self.device = torch.device(f"cuda:{self.num_gpu}")
            
        else:
            print("Using CPU...")
            self.device = torch.device("cpu")
            
        torch.set_num_threads(self.torch_threads)
        
        self.create_env()

        if self.model_dir is not None and self.algorithm_name == "mast":  # restore model
            self.restore(model_dir = self.exp_args["model_dir"])
            
        timer_cancel(timer = timer)
    
    def create_env(self):
        
        if self.curriculum_level > 0:
            saved_state = {
                "actor_state_dict": self.policy.actor.state_dict(),
                "critic_state_dict": self.policy.critic.state_dict(),
                "actor_optimizer_state_dict": self.policy.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.policy.critic_optimizer.state_dict(),
            }
        
        if self.use_render:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(
                self.env_name, 
                self.seed, 
                self.env_info
            )
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                env_name = self.env_name,
                seed = self.seed,
                n_threads = self.n_rollout_threads,
                env_args = self.env_info,
            )
            self.eval_envs = (
                make_eval_env(
                env_name = self.env_name,
                seed = self.seed,
                n_threads = self.n_eval_rollout_threads,
                env_args = self.env_info,
                )
                if self.use_eval
                else None
            )
            
        self.num_agents, self.n_actions_no_attack, self.num_objects = get_num_agents(envs = self.envs)
        
        observation_space = self.envs.observation_space[0]
        share_observation_space = self.envs.share_observation_space[0] 
        action_space = self.envs.action_space[0]

        print("share_observation_space: ", observation_space)
        print("observation_space: ", share_observation_space)
        print("action_space: ", action_space)
        
        if self.algorithm_name == "mast":
            from algorithms.mast.mast import Mast as TrainAlgo
            from algorithms.mast.algorithm.MastPolicy import MastPolicy as Policy
        elif self.algorithm_name == "mat":
            from algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
            from algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy      
        else:
            from algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        self.num_objects = sum(map(int, re.findall('\d+', self.map_name)))


        self.policy = Policy(
            args = self.policy_args, 
            obs_space = observation_space, 
            share_obs_space = share_observation_space, 
            act_space = action_space, 
            num_agents = self.num_agents, 
            num_objects = self.num_objects, 
            n_actions_no_attack = self.n_actions_no_attack,
            device = self.device
        )
        if self.curriculum_level > 0:
            self.policy.actor.load_state_dict(saved_state["actor_state_dict"])
            self.policy.critic.load_state_dict(saved_state["critic_state_dict"])
            self.policy.actor_optimizer.load_state_dict(saved_state["actor_optimizer_state_dict"])
            self.policy.critic_optimizer.load_state_dict(saved_state["critic_optimizer_state_dict"])

        # algorithm
        if self.algorithm_name == "mat":
            self.trainer = TrainAlgo(
                self.args, 
                self.policy, 
                self.num_agents, 
                device = self.device
            )
        elif self.algorithm_name == "mast":
            self.trainer = TrainAlgo(
                self.args, 
                self.policy, 
                self.num_agents, 
                device = self.device
            )
        else:
            self.trainer = TrainAlgo(
                self.args, 
                self.policy, 
                device = self.device
            )

        self.buffer = SharedReplayBuffer(
            self.args,
            self.num_agents,
            observation_space,
            share_observation_space,
            action_space
        )

        self.logger = LOGGER_REGISTRY[self.exp_args["env_name"]](
            exp_args = self.exp_args,
            algo_args = self.algo_args,
            env_args = self.env_args,  
            wandb = self.wandb if self.use_wandb and not self.use_render else None,
        )
        
        
    def curriculum_manager(self, eval_win_rate):
        self.weighted_win_rate = (1 - self.decay_rate) * self.weighted_win_rate + self.decay_rate * eval_win_rate
        if self.weighted_win_rate >= self.threshold and self.curriculum_level < len(self.curriculum_envs) - 1 and self.stack > self.stack_threshold:   
            self.curriculum_level += 1
            self.map_name = self.curriculum_envs[self.curriculum_level] 
            self.exp_args["map_name"] = self.map_name
            self.env_info = {"map_name":self.map_name, "algorithm_name":self.algorithm_name}
            self.create_env()
            self.stack = 0
        else:
            self.stack += 1

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.obs[-1]),
            np.concatenate(self.buffer.rnn_states[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
            np.concatenate(self.buffer.available_actions[-1])
        )
        
    
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    

    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)   
        self.buffer.after_update()   
        return train_infos

    def save(self, episode=0):
        """Save policy's actor and critic networks."""
        save_directory = f"{self.save_dir}/{episode}"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.save(save_dir = save_directory)
        elif self.algorithm_name == "mappo"  or self.algorithm_name == "mast":
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(save_directory) + "/actor.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(save_directory) + "/critic.pt")
        else:
            raise NotImplementedError

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.restore(model_dir)
        else:
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
            self.policy.actor.load_state_dict(policy_actor_state_dict)

            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)