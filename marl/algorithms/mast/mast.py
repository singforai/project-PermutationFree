import torch

import numpy as np
import torch.nn as nn

from utils.valuenorm import ValueNorm
from algorithms.utils.util import check
from utils.util import get_gard_norm, huber_loss, mse_loss

class Mast():
    def __init__(self, args, policy, num_agents, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        self.policy = policy
        self.num_agents = num_agents
        
        self.clip_param = args["algo"]["clip_param"]
        self.ppo_epoch = args["algo"]["ppo_epoch"]
        self.num_mini_batch = args["algo"]["num_mini_batch"]
        self.data_chunk_length = args["algo"]["data_chunk_length"]
        self.value_loss_coef = args["algo"]["value_loss_coef"]
        self.entropy_coef = args["algo"]["entropy_coef"]
        self.max_grad_norm = args["algo"]["max_grad_norm"]       
        self.huber_delta = args["algo"]["huber_delta"]
        
        self.use_recurrent_policy = args["model"]["use_recurrent_policy"]
        self._use_max_grad_norm = args["algo"]["use_max_grad_norm"]
        self._use_clipped_value_loss = args["algo"]["use_clipped_value_loss"]
        self._use_huber_loss = args["algo"]["use_huber_loss"]
        self._use_popart = args["train"]["use_popart"]
        self._use_valuenorm = args["algo"]["use_valuenorm"]
        self._use_value_active_masks = args["algo"]["use_value_active_masks"]
        self._use_policy_active_masks = args["algo"]["use_policy_active_masks"]
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
    
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def to_single_np(self, input):
        reshape_input = input.reshape(-1, self.num_agents, *input.shape[1:])
        return reshape_input[:, 0, ...]
            
    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss            

    def network_update(self, sample):
        """
        Arguements
            - share_obs_batch
            - obs_batch
            - rnn_states_batch
            - rnn_states_critic_batch
            - actions_batch
            - value_preds_batch
            - return_batch
            - masks_batch
            - active_masks_batch
            - old_action_log_probs_batch
            - adv_targ
            - available_actions_batch
        """
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            visible_masking_batch
        ) = sample

        adv_targ = check(adv_targ).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)

        mini_batch_size, num_objects, feature_dim = obs_batch.shape
        mini_batch_size, num_objects, recurrent_N, hidden_size = rnn_states_batch.shape
        
        obs_batch = obs_batch.reshape(mini_batch_size * num_objects, feature_dim)
        rnn_states_batch = rnn_states_batch.reshape(mini_batch_size * num_objects, recurrent_N, hidden_size)
        actions_batch = actions_batch.reshape(mini_batch_size * self.num_agents, 1)
        value_preds_batch = value_preds_batch.reshape(mini_batch_size * self.num_agents, 1)
        return_batch = return_batch.reshape(mini_batch_size * self.num_agents, 1)
        masks_batch = masks_batch.reshape(mini_batch_size * num_objects, 1)
        active_masks_batch  = active_masks_batch.reshape(mini_batch_size * self.num_agents, 1)
        old_action_log_probs_batch = old_action_log_probs_batch.reshape(mini_batch_size * self.num_agents, 1)
        adv_targ = adv_targ.reshape(mini_batch_size * self.num_agents, 1)
        available_actions_batch = available_actions_batch.reshape(mini_batch_size * self.num_agents, -1)

        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            obs=obs_batch,
            rnn_states = rnn_states_batch,
            visible_masking = visible_masking_batch,
            action=actions_batch,
            masks=masks_batch,
            available_actions=available_actions_batch,
            active_masks=active_masks_batch,
        )
            
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        
        policy_loss = policy_action_loss
        
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.optimizer.zero_grad()
        ((value_loss * self.value_loss_coef) + (policy_loss - dist_entropy * self.entropy_coef)).backward()
        
        
        if self._use_max_grad_norm:
            model_grad_norm = nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        else:
            model_grad_norm = get_gard_norm(self.policy.model.parameters())
            
        self.policy.optimizer.step()

        return value_loss, policy_loss, dist_entropy, imp_weights, model_grad_norm
        
    def train(self, buffer):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['ratio'] = 0
        train_info["model_grad_norm"] = 0
        
        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = buffer.recurrent_transformer_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            else:
                data_generator = buffer.forward_transformer_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, policy_loss, dist_entropy, imp_weights, model_grad_norm = self.network_update(sample)
                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["ratio"] += imp_weights.mean()
                train_info["model_grad_norm"] += model_grad_norm
                
        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
                
        return train_info
        
    def prep_training(self):
        self.policy.model.train()

    def prep_rollout(self):
        self.policy.model.eval()
    