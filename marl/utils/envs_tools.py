import os
import torch 
import random 

import numpy as np

from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def set_seed(seed):
    """Seed the program."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def make_train_env(env_name, seed, n_threads, env_args):
    """Make env for training."""
    def get_env_fn(rank):
        def init_env():
            if env_name == "smacv2":
                from envs.smacv2.smacv2_env import SMACv2Env
                env = SMACv2Env(env_args)
            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
            env.seed(seed + rank * 1000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_eval_env(env_name, seed, n_threads, env_args):
    """Make env for evaluation."""
    if env_name == "dexhands":  # dexhands does not support running multiple instances
        raise NotImplementedError

    def get_env_fn(rank):
        def init_env():
            if env_name == "smacv2":
                from envs.smacv2.smacv2_env import SMACv2Env
                env = SMACv2Env(env_args)
            else:
                print("Can not support the " + env_name + "environment.")
                raise NotImplementedError
            env.seed(seed * 50000 + rank * 10000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_render_env(env_name, seed, env_args):
    """Make env for rendering."""
    manual_render = True  # manually call the render() function
    manual_expand_dims = True  # manually expand the num_of_parallel_envs dimension
    manual_delay = True  # manually delay the rendering by time.sleep()
    env_num = 1  # number of parallel envs
    if env_name == "smacv2":
        from envs.smacv2.smacv2_env import SMACv2Env
        env = SMACv2Env(args=env_args)
        manual_render = False
        manual_delay = False
        env.seed(seed * 60000)
    else:
        print("Can not support the " + env_name + "environment.")
        raise NotImplementedError
    return env, manual_render, manual_expand_dims, manual_delay, env_num


def get_num_agents(env, env_args, envs):
    """Get the number of agents in the environment."""
    if env == "smacv2":
        return envs.n_agents
    else:
        NotImplementedError
