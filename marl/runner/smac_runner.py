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

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, exp_args, algo_args, env_args):
        super(SMACRunner, self).__init__(exp_args, algo_args, env_args)

    def run(self):
        
        start = time.time()
        episodes = int(float(self.num_env_steps) // self.episode_length // self.n_rollout_threads)
        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        timer = timer_start()
        self.warmup()  
        timer_cancel(timer = timer)
        
        for episode in range(episodes):
            timer = timer_start()
            
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
                
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            
            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))


                # battles_won = []
                # battles_game = []
                # incre_battles_won = []
                # incre_battles_game = []                    

                # for i, info in enumerate(infos):
                #     if 'battle_won' in info[0].keys() and 'battle_game' in info[0].keys():
                #         battles_won.append(int(info[0]['battle_won']))
                #         incre_battles_won.append(info[0]['battle_won']-last_battles_won[i])
                #         battles_game.append(info[0]['battle_game'])
                #         incre_battles_game.append(info[0]['battle_game']-last_battles_game[i])

                # incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                # print("incre win rate is {}.".format(incre_win_rate))
                # if self.use_wandb:
                #     wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                # last_battles_game = battles_game
                # last_battles_won = battles_won

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape)) 
                train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
                self.log_train(train_infos, total_num_steps)

            if self.save_model and (episode % self.save_interval == 0):
                self.save(episode = episode)
                    
            # eval
            if self.use_eval and episode % self.eval_interval == 0:
                self.eval(total_num_steps)
            
            timer_cancel(timer = timer)
        
    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        if self.algorithm_name == "mast":
            self.buffer.obs[0] = obs.copy()
            self.buffer.available_actions[0] = available_actions.copy()
        else:
            self.buffer.share_obs[0] = share_obs.copy()
            self.buffer.obs[0] = obs.copy()
            self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        if self.algorithm_name == "mast":
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer.policy.get_actions(
                np.concatenate(self.buffer.obs[step]), 
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.masks[step]),
                np.concatenate(self.buffer.available_actions[step])
            )
        else:
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]), 
                np.concatenate(self.buffer.obs[step]), # rollout x num_agent x feature
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
        if self.algorithm_name != "mast":
            rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        else:
            rnn_states_critic = None
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        (
            obs,  # (n_threads, n_agents, obs_dim)
            share_obs,  # (n_threads, n_agents, share_obs_dim)
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
        if self.algorithm_name != "mast":
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

    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps = 0):
        eval_battles_won = 0
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            if self.algorithm_name == "mast":
                eval_actions, eval_rnn_states = self.trainer.policy.act(
                    np.concatenate(eval_obs),
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_masks),
                    np.concatenate(eval_available_actions),
                    deterministic=True
                )
            
            else:
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
            
            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    if eval_infos[eval_i][0]['battle_won']:
                        eval_battles_won += 1
                        
            if eval_episode > self.eval_episodes: 
                eval_win_rate = eval_battles_won/eval_episode
                print(f"eval win rate is {eval_win_rate}.")
                if self.use_wandb and (self.use_render == False):
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                break

    def close(self):
        """Close environment, writter, and logger."""

        self.envs.close()
        if self.use_eval:
            self.eval_envs.close()

        if self.use_wandb and not self.use_render:
            self.wandb.finish()
