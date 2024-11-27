import time
from functools import reduce
import numpy as np

class SMACv2Logger():
    def __init__(self, exp_args, algo_args, env_args, num_agents, wandb):
        self.exp_args = exp_args
        self.algo_args = algo_args
        self.env_args = env_args
        self.num_agents = num_agents
        self.wandb = wandb
        
        self.game_key = "battles_game"
        self.win_key = "battles_won"

    def init(self, episodes):
        self.episodes = episodes
        self.episode_lens = []
        self.one_episode_len = np.zeros(
            self.algo_args["train"]["n_rollout_threads"], dtype=np.int32
        )
        self.last_battles_game = np.zeros(
            self.algo_args["train"]["n_rollout_threads"], dtype=np.float32
        )
        self.last_battles_won = np.zeros(
            self.algo_args["train"]["n_rollout_threads"], dtype=np.float32
        )
        
    def episode_init(self, episode):
        """Initialize the logger for each episode."""
        self.episode = episode


    def per_step(self, dones, infos):
        self.infos = infos
        self.one_episode_len += 1
        done_env = np.all(dones, axis=1)
        for i in range(self.algo_args["train"]["n_rollout_threads"]):
            if done_env[i]:
                self.episode_lens.append(self.one_episode_len[i].copy())
                self.one_episode_len[i] = 0

    def episode_log(
        self, total_num_steps, train_infos, buffer
    ):
        self.total_num_steps = total_num_steps
        print(
            "Env {} Task {} Algo {} Exp {} updates {}/{} episodes, total num timesteps.".format(
                self.exp_args["env_name"],
                self.exp_args["map_name"],
                self.exp_args["algorithm_name"],
                self.exp_args["experiment_name"],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.env_args["num_env_steps"],
            )
        )
        battles_won = [0 for _ in range(self.algo_args["train"]["n_rollout_threads"])]    
        battles_game = [0 for _ in range(self.algo_args["train"]["n_rollout_threads"])]   
        incre_battles_won = [0 for _ in range(self.algo_args["train"]["n_rollout_threads"])]   
        incre_battles_game = [0 for _ in range(self.algo_args["train"]["n_rollout_threads"])]   

        for i, info in enumerate(self.infos):
            if self.win_key in info[0].keys():
                battles_won[i] = info[0][self.win_key]
                incre_battles_won[i] = (
                    info[0][self.win_key] - self.last_battles_won[i]
                )
            if self.game_key in info[0].keys():
                battles_game[i] = info[0][self.game_key]
                incre_battles_game[i] = (
                    info[0][self.game_key] - self.last_battles_game[i]
                )

        incre_win_rate = (
            np.sum(incre_battles_won) / np.sum(incre_battles_game)
            if np.sum(incre_battles_game) > 0
            else 0.0
        )
        
        self.last_battles_game = battles_game
        self.last_battles_won = battles_won

        average_episode_len = (
            np.mean(self.episode_lens) if len(self.episode_lens) > 0 else 0.0
        )
        
        train_infos["average_episode_len"] = average_episode_len
        train_infos['dead_ratio'] = 1 - buffer.active_masks[1:].sum() / reduce(lambda x, y: x*y, list(buffer.active_masks[1:].shape)) 
        train_infos["incre_win_rate"] = incre_win_rate
        train_infos["average_episode_len"] = average_episode_len    
        train_infos["average_step_rewards"] = np.mean(buffer.rewards)
        
        self.episode_lens = []

        self.log_train(train_infos, total_num_steps)

    def eval_init(self, total_num_steps):
        """Initialize the logger for evaluation."""
        
        self.total_num_steps = total_num_steps
        
        self.eval_episode_rewards = []
        self.one_episode_rewards = []
        for _ in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards.append([])
            
        self.eval_battles_won = 0
        
    def eval_per_step(self, eval_rewards, eval_infos):
        """Log evaluation information per step."""

        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
        self.eval_infos = eval_infos

    def eval_thread_done(self, tid):
        
        """Log evaluation information."""
        self.eval_episode_rewards.append(
            np.sum(self.one_episode_rewards[tid], axis=0)
        )
        self.one_episode_rewards[tid] = []
        
        if self.eval_infos[tid][0]["battle_won"] == True:
            self.eval_battles_won += 1

    def eval_log(self, eval_episode):
        eval_mean_rewards = np.array(self.eval_episode_rewards).mean()
        eval_max_rewards = np.array(self.eval_episode_rewards).mean(axis = 1).max()
        
        eval_win_rate = self.eval_battles_won / eval_episode
        eval_env_infos = {
            "eval_average_episode_rewards": eval_mean_rewards,
            "eval_max_episode_rewards": eval_max_rewards,
            "eval_win_rate": eval_win_rate,
        }
        print(
            "Evaluation win rate is {}, evaluation average episode reward is {}.\n".format(
                eval_win_rate, eval_mean_rewards
            )
        )
        self.log_env(eval_env_infos, self.total_num_steps)
        
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.wandb is not None:
                self.wandb.log({k: v}, step=total_num_steps)
                
    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if self.wandb is not None:
                self.wandb.log({k: np.mean(v)}, step=total_num_steps)
