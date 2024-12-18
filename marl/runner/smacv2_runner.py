import os 
import time
import wandb
import threading
import numpy as np
from functools import reduce
import torch
from runner.base_runner import Runner

from utils.util import timer_start, timer_cancel

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACv2Runner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, exp_args, algo_args, env_args):
        super().__init__(exp_args, algo_args, env_args)
 
    def run(self):
        
        episodes = int(float(self.num_env_steps) // self.episode_length // self.n_rollout_threads)
        self.init_env()
        for episode in range(episodes):
            timer = timer_start()
            self.logger.episode_init(episode)
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
                
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                # Obser reward and next obs
                share_obs, obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 

                # insert data into buffer
                self.insert(data)

                self.logger.per_step(
                    dones,
                    infos
                )

            # compute return and update network
            self.compute()
            
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            
            # log information
            if episode % self.log_interval == 0:    
                self.logger.episode_log(
                    total_num_steps,
                    train_infos,
                    self.buffer,
                )

            if self.save_model and (episode % self.save_interval == 0):
                self.save(episode = episode)

            # eval
            if self.use_eval and episode % self.eval_interval == 0:
                eval_win_rate = self.eval(total_num_steps)
                if self.use_curriculum:
                    self.curriculum_manager(eval_win_rate = eval_win_rate)
                    self.init_env()
            
            timer_cancel(timer = timer)
        
        self.save(episode = episode)
        
    def init_env(self):
        timer = timer_start()
        self.warmup()  
        timer_cancel(timer = timer)
        self.logger.init()
        
    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]), 
            np.concatenate(self.buffer.obs[step]), 
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            np.concatenate(self.buffer.available_actions[step])
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        (
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            obs,  # (n_threads, n_agents, obs_dim)            
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: list, shape: (n_threads, n_agents)
            available_actions,  # (n_threads, ) of None or (n_threads, n_agents, action_number)
            values,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
            actions,  # (n_threads, n_agents, action_dim)
            action_log_probs,  # (n_threads, n_agents, action_dim)
            rnn_states,  # (n_threads, n_agents, dim)
            rnn_states_critic,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
        ) = data
        
        dones_env = np.all(dones, axis=1)
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        
        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array(
            [
                [
                    [0.0]
                    if "bad_transition" in info[agent_id].keys()
                    and info[agent_id]["bad_transition"] == True
                    else [1.0]
                    for agent_id in range(self.num_agents)
                ]
                for info in infos
            ]
        )
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions)
    
    @torch.no_grad()
    def eval(self, total_num_steps = 0):

        self.logger.eval_init(total_num_steps, curriculum_level = self.curriculum_level)
        
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            
            eval_actions, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_share_obs),
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                np.concatenate(eval_available_actions),
                deterministic=True
            )
                    
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)

            self.logger.eval_per_step(
                eval_rewards,
                eval_infos
            )
            
            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)
                        
            if eval_episode >= self.eval_episodes: 
                return self.logger.eval_log(
                    eval_episode
                )  

    def close(self):
        """Close environment, writter, and logger."""

        self.envs.close()
        if self.use_eval:
            self.eval_envs.close()

        if self.use_wandb and not self.use_render:
            self.wandb.finish()