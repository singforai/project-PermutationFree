import torch
import numpy as np
import torch.nn.functional as F
from utils.util import get_shape_from_obs_space, get_shape_from_act_space

def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    """
    (episode_length / n_rollout_threads / num_agents / 393)
    => 
    (n_rollout_threads / num_agents / episode_length / 393)
    => 
    (n_rollout_threads * num_agents * episode_length / 393)
    """
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])

def _flatten_v3(T, N, agent_num, x):
    return x.reshape(T * N * agent_num, *x.shape[3:])


def _cast_v3(x):
    return x.transpose(1, 0, 2, 3).reshape(-1, *x.shape[2:])


def _shuffle_agent_grid(x, y): # Functions for setting the order of agents
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols

class ObjectSharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents, num_objects, obs_space, cent_obs_space, act_space):
        self.episode_length = args["train"]["episode_length"]
        self.n_rollout_threads = args["train"]["n_rollout_threads"]
        self.hidden_size = args["model"]["hidden_size"]
        self.recurrent_N = args["model"]["recurrent_N"]
        self.gamma = args["algo"]["gamma"]
        self.gae_lambda = args["algo"]["gae_lambda"]
        self._use_gae = args["algo"]["use_gae"]
        self._use_popart = args["train"]["use_popart"]
        self._use_valuenorm = args["algo"]["use_valuenorm"]
        self._use_proper_time_limits = args["train"]["use_proper_time_limits"]
        self.algo = args["algorithm_name"]
        self.num_agents = num_agents
        self.num_objects = num_objects

        self.act_space_n = act_space.n
        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.act_shape = get_shape_from_act_space(act_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]
            
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_objects, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.num_objects, self.recurrent_N, self.hidden_size),
            dtype=np.float32)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, self.num_agents, act_space.n),
                                             dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)
        self.obs_shape = (*obs_shape,)[0]
        self.share_obs_shape = (*share_obs_shape,)[0]

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, self.num_objects, 1), dtype=np.float32)
        
        self.bad_masks = np.ones((self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.active_masks = np.ones_like(self.bad_masks)
        
        self.visible_masking =np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_objects, self.num_objects),dtype=np.float32)
        
        self.step = 0
        
    def insert(self, obs, visible_masking, rnn_states_actor, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.obs[self.step + 1] = obs.copy()
        self.visible_masking[self.step + 1] = visible_masking.copy()
        self.rnn_states[self.step + 1] = rnn_states_actor.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()
            
        self.visible_masking[0] = self.visible_masking[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1, :, : self.num_agents] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1, :, : self.num_agents]
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1, :, : self.num_agents] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1, :, : self.num_agents] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1, :, : self.num_agents] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1, :, : self.num_agents] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.active_masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1, :, : self.num_agents] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * \
                                self.masks[step + 1, :, : self.num_agents] - self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1, :, : self.num_agents] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.self.masks[step + 1, :, : self.num_agents] + self.rewards[step]

    def recurrent_transformer_generator(self, advantages, num_mini_batch, data_chunk_length):
        
        episode_length, n_rollout_threads, self.num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length
        num_data_chunks = batch_size // data_chunk_length # data chunk 개수
        mini_batch_size = num_data_chunks // num_mini_batch # mini-batch 하나당 데이터 개수

        assert n_rollout_threads * episode_length >= data_chunk_length, (
            f"PPO requires the number of processes ({n_rollout_threads}) * episode length ({episode_length}) "
            "to be greater than or equal to the number of "
            "data chunk length ({data_chunk_length})."
        )
        
        rand = torch.randperm(num_data_chunks).numpy() 
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)] 

        actions = _cast_v3(self.actions)
        advantages = _cast_v3(advantages)
        action_log_probs = _cast_v3(self.action_log_probs)
        
        masks = _cast_v3(self.masks[:-1])
        policy_obs = _cast_v3(self.obs[:-1])     
        returns = _cast_v3(self.returns[:-1])
        value_preds = _cast_v3(self.value_preds[:-1])
        active_masks = _cast_v3(self.active_masks[:-1])
        if self.available_actions is not None:
            available_actions = _cast_v3(self.available_actions[:-1])
            
        rnn_states = self.rnn_states[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.rnn_states.shape[2:])
        
        for mini_batch in sampler:
            obs_batch = []
            rnn_states_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            
            for step in mini_batch:
                ind = step * data_chunk_length
                
                obs_batch.append(policy_obs[ind : ind + data_chunk_length])
                actions_batch.append(actions[ind : ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind : ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind : ind + data_chunk_length])
                return_batch.append(returns[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind : ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind : ind + data_chunk_length])
                adv_targ.append(advantages[ind : ind + data_chunk_length])

                rnn_states_batch.append(rnn_states[ind])
            
            L, N = data_chunk_length, mini_batch_size
            
            """
            size when num_mini_batch is 1
            (num_rollout * episode_length // data_chunk_length, data_chunk_length, agent_num, obs_Dim)
            => (data_chunk_length, num_rollout * episode_length // data_chunk_length, agent_num, obs_Dim)
            """
            obs_batch = np.stack(obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)
            
            """
            size when num_mini_batch is 1
            (num_rollout * episode_length // data_chunk_length, agent_num, reccurent_N, hidden_size)
            => (num_rollout * episode_length // data_chunk_length * agent_num, reccurent_N, hidden_size)
            """
            rnn_states_batch = np.stack(rnn_states_batch).reshape(
                N * self.num_objects, *self.rnn_states.shape[3:]
            )
            
            obs_batch = _flatten_v3(L, N, self.num_objects, obs_batch)
            actions_batch = _flatten_v3(L, N, self.num_agents, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten_v3(L, N, self.num_agents, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten_v3(L, N, self.num_agents, value_preds_batch)
            return_batch = _flatten_v3(L, N, self.num_agents, return_batch)
            masks_batch = _flatten_v3(L, N, self.num_objects, masks_batch)
            active_masks_batch = _flatten_v3(L, N, self.num_agents, active_masks_batch)
            old_action_log_probs_batch = _flatten_v3(L, N, self.num_agents, old_action_log_probs_batch)
            adv_targ = _flatten_v3(L, N, self.num_agents, adv_targ)
            
            yield obs_batch, rnn_states_batch, actions_batch, value_preds_batch, \
            return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch
        
        
    def forward_transformer_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        
        episode_length, n_rollout_threads, self.num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, self.num_agents,
                          n_rollout_threads * episode_length * self.num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        
        rand = torch.randperm(batch_size).numpy() 
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)] 
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        visible_masking = self.visible_masking[:-1].reshape(-1, *self.visible_masking.shape[-2:])
        actions = self.actions.reshape(-1, *self.actions.shape[-2:])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[-2:])
        value_preds = self.value_preds[:-1].reshape(-1, self.num_agents, 1)
        returns = self.returns[:-1].reshape(-1, self.num_agents, 1)
        masks = self.masks[:-1].reshape(-1, self.num_objects, 1)
        active_masks = self.active_masks[:-1].reshape(-1, self.num_agents, 1)
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[-2:])
        advantages = advantages.reshape(-1, self.num_agents, 1)
        for indices in sampler:
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            visible_masking_batch = visible_masking[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield obs_batch, rnn_states_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
                  adv_targ, available_actions_batch, visible_masking_batch