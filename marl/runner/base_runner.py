import os 
import re
import wandb
import torch
import socket
import setproctitle

import numpy as np
from gym import spaces

from utils.util import timer_start, timer_cancel

from utils.mast_shared_buffer import ObjectSharedReplayBuffer
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
        
        self.map_name: str = self.exp_args["map_name"]
        self.env_name: str = self.exp_args["env_name"]
        self.log_dir: str = self.exp_args["log_dir"]
        self.model_dir: str = self.exp_args["model_dir"]
        self.user_name: str = self.exp_args["user_name"]
        self.group_name: str = self.exp_args["group_name"]
        self.experiment_name: str = self.exp_args["experiment_name"] 
        self.algorithm_name: str = self.exp_args["algorithm_name"]
        
        # algo args
        self.use_eval: bool = self.algo_args["eval"]["use_eval"] 
        self.use_linear_lr_decay: bool = self.algo_args["train"]["use_linear_lr_decay"]
           
        self.episode_length: int = self.algo_args["train"]["episode_length"]    
        self.n_rollout_threads: int = self.algo_args["train"]["n_rollout_threads"]   
        self.n_eval_rollout_threads : int = self.algo_args["eval"]["n_eval_rollout_threads"]  
        self.recurrent_N: int = self.algo_args["model"]["recurrent_N"]
        self.hidden_size: int = self.algo_args["model"]["hidden_size"]  
        self.eval_episodes: int = self.algo_args["eval"]["eval_episodes"]
        
        # env args
        self.num_env_steps: int = self.env_args["num_env_steps"]
        env_info: Dict[str, Any] = {"map_name":self.map_name, "algorithm_name":self.algorithm_name}
        
        # united args
        self.policy_args: Dict[str, Any] = {**self.exp_args, **self.algo_args["train"], **self.algo_args["model"], **self.algo_args["algo"]}
        self.args: Dict[str, Any] = {**self.exp_args, **self.algo_args, **self.env_args}
        
        set_seed(seed = self.seed)
        if not self.use_render:
            self.run_dir, self.save_dir = init_dir(
                env_name = self.env_name,
                map_name = self.map_name,
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
                env_info
            )
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                env_name = self.env_name,
                seed = self.seed,
                n_threads = self.n_rollout_threads,
                env_args = env_info,
            )
            self.eval_envs = (
                make_eval_env(
                env_name = self.env_name,
                seed = self.seed,
                n_threads = self.n_eval_rollout_threads,
                env_args = env_info,
                )
                if self.use_eval
                else None
            )
            
        self.num_agents = get_num_agents(
            env = self.env_name, 
            env_args = self.env_args, 
            envs = self.envs
        )
        
        if self.use_gpu and torch.cuda.is_available():
            print("Using GPU...")
            self.device = torch.device(f"cuda:{self.num_gpu}")
            
        else:
            print("Using CPU...")
            self.device = torch.device("cpu")
        torch.set_num_threads(self.torch_threads)

        if self.use_wandb and not self.use_render:
            self.wandb = wandb.init(
                config={**self.exp_args, **self.algo_args, **self.env_args},
                project= self.env_name + "_" + self.map_name,     
                notes=socket.gethostname(),
                entity=self.user_name,
                name="-".join([self.algorithm_name, self.experiment_name, "seed" + str(self.seed)]),
                group=self.group_name,
                dir=str(self.run_dir),
                job_type="training",
            )
        
        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        if self.algorithm_name == "mast":
            from algorithms.mast.mast import Mast as TrainAlgo
            from algorithms.mast.algorithm.MastPolicy import MastPolicy as Policy
        else:
            from algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        
        observation_space = self.envs.observation_space[0]
        share_observation_space = self.envs.share_observation_space[0] 
        action_space = self.envs.action_space[0]
        self.num_objects = sum(map(int, re.findall('\d+', self.map_name)))
        
        # policy network
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy = Policy(
                self.policy_args, 
                observation_space, 
                share_observation_space, 
                action_space, 
                self.num_agents, 
                device = self.device
            )
        elif self.algorithm_name == "mast":
            self.policy = Policy(
                args = self.policy_args, 
                obs_space = observation_space, 
                cent_obs_space = share_observation_space, 
                act_space = action_space, 
                num_agents = self.num_agents, 
                num_objects = self.num_objects, 
                device = self.device
            )
        else:
            self.policy = Policy(
                args = self.policy_args, 
                obs_space = observation_space, 
                cent_obs_space = share_observation_space, 
                act_space = action_space, 
                num_agents = self.num_agents, 
                device = self.device
            )

        if self.model_dir is not None:  # restore model
            self.restore()

        # algorithm

        # algorithm
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
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
        
        if self.algorithm_name == "mast":
            self.buffer = ObjectSharedReplayBuffer(
                self.args,
                self.num_agents,
                self.num_objects,
                observation_space,
                share_observation_space,
                self.envs.action_space[0]
            )
        else:
            self.buffer = SharedReplayBuffer(
                self.args,
                self.num_agents,
                observation_space,
                share_observation_space,
                self.envs.action_space[0]
            )
            
        self.logger = LOGGER_REGISTRY[self.exp_args["env_name"]](
            exp_args = self.exp_args,
            algo_args = self.algo_args,
            env_args = self.env_args,  
            num_agents = self.num_agents,
            wandb = self.wandb if self.use_wandb and not self.use_render else None,
        )
            
        timer_cancel(timer = timer)

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
        elif self.algorithm_name == "rmappo" or self.algorithm_name == "mappo" or self.algorithm_name == "tizero"  or self.algorithm_name == "mast":
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
            if not self.args.use_render:
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