import torch
import itertools
from utils.util import update_linear_schedule
from utils.util import get_shape_from_obs_space

from algorithms.utils.util import check

from algorithms.mast.algorithm.setransformer import  Actor, Critic

class MastPolicy:
    def __init__(
        self, args, obs_space, share_obs_space, act_space, num_agents, num_objects, n_actions_no_attack, device=torch.device("cpu")
    ):
        self.device = device
        self.lr = args["lr"]
        self.critic_lr = args["critic_lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]
        
        self.hidden_size: int = args["hidden_size"]
        self.check_permutation_free = args["check_permutation_free"]    

        self.num_agents: int = num_agents
        self.num_objects: int = num_objects  
        self.n_actions_no_attack: int = n_actions_no_attack
        
        self.obs_shape = get_shape_from_obs_space(obs_space)[0]
        self.share_obs_shape = get_shape_from_obs_space(share_obs_space)[0]
        self.action_dim = act_space.n
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        self._use_recurrent_policy = args["use_recurrent_policy"]
        self._use_policy_active_masks = args["use_policy_active_masks"]
        self.input_dim = self.obs_shape // self.num_objects
        
        self.threadshold = 1e-7
        
        self.actor = Actor(
            args = args, 
            obs_space = self.obs_shape, 
            num_agents = self.num_agents,
            num_objects = self.num_objects,
            n_actions_no_attack = self.n_actions_no_attack,
            device = self.device
        )
        self.critic = Critic(
            args = args,
            obs_space = self.share_obs_shape, 
            num_agents = self.num_agents,
            num_objects = self.num_objects,
            device = device
        )
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr, 
            eps=self.opti_eps,
            weight_decay=self.weight_decay
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay
        )
        
    def lr_decay(self, episode, episodes):
        """
        If use_linear_lr_decay is True, the learning rate decreases linearly

        Arguement:
            - episode           | int
            - episodes          | int
        
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        
    def get_actions(self, share_obs, obs, rnn_states, rnn_states_critic, masks, available_actions=None, deterministic=False):
        """
        A function to sample the next action during the sampling process

        Arguement:
            - cent_obs          | np.ndarray (n_rollout_threads * num_agents, share_obs_dim)
            - obs               | np.ndarray (n_rollout_threads * num_agents, obs_dim)
            - rnn_states        | np.ndarray (n_rollout_threads * num_agents, recurrent_N, hidden_size)
            - rnn_states_critic | np.ndarray (n_rollout_threads * num_agents, recurrent_N, hidden_size)
            - masks             | np.ndarray (n_rollout_threads * num_agents, 1)
            - available_actions | np.ndarray (n_rollout_threads * num_agents, action_space)
            - deterministic     | bool

        return:
            - values            | tensor (n_rollout_threads * num_agents , 1)
            - actions           | tensor (n_rollout_threads * num_agents , 1)
            - action_log_probs  | tensor (n_rollout_threads * num_agents , 1)
            - rnn_states        | tensor (n_rollout_threads * num_agents , recurrent_N, hidden_size)
            - rnn_states_critic | tensor (n_rollout_threads * num_agents , recurrent_N, hidden_size)
        """
        share_obs = check(share_obs).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            
        actions, action_log_probs, rnn_states = self.actor(
            obs,
            rnn_states,
            masks,
            available_actions,
            deterministic,
        )

        values, rnn_states_critic = self.critic(
            share_obs,
            rnn_states_critic,
            masks,
        )
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    def get_values(self, share_obs, obs, rnn_states, rnn_states_critic, masks, available_actions=None):
        share_obs = check(share_obs).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            
        values, _ = self.critic(
            share_obs,
            rnn_states_critic,
            masks,
        )
        
        return values
    
    def evaluate_actions(self, share_obs, obs, rnn_states, rnn_states_critic,
                          action, masks, available_actions = None, active_masks = None):
        """
        A function to calculate the importance weight and value loss between the updated network 
        from training and the network used for sampling
        
        Arguement:
            - obs               | np.ndarray (data_chunk_length * mini_batch_size * num_agents ,  obs_dim)
            - rnn_states        | np.ndarray (mini_batch_size * num_agents , recurrent_N , hidden_size)
            - rnn_states_critic | np.ndarray (mini_batch_size * num_agents , recurrent_N , hidden_size)
            - action            | np.ndarray (data_chunk_length * mini_batch_size * num_agents ,  1)
            - masks             | np.ndarray (data_chunk_length * mini_batch_size * num_agents ,  1)
            - critic_masks_batch| np.ndarray (data_chunk_length * mini_batch_size * num_agents ,  1)
            - available_actions | np.ndarray (data_chunk_length * mini_batch_size * num_agents ,  action_space)
            - active_masks      | np.ndarray (data_chunk_length * mini_batch_size * num_agents ,  1)
            
        return:
            - values            | tensor (data_chunk_length * mini_batch_size * num_agents ,  1)
            - action_log_probs  | tensor (data_chunk_length * mini_batch_size * num_agents ,  1)
            - dist_entropy      | tensor float
        """
        share_obs = check(share_obs).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            
        obs = obs.reshape(obs.shape[0], self.num_objects, self.input_dim)
        share_obs = share_obs.reshape(share_obs.shape[0], self.num_objects, self.input_dim)
        
        x = self.actor.base(obs)
        object_key = self.actor._feature_SAB(x)
        x = self.actor._feature_PMA(object_key)
        x = x.mean(dim = 1)
        if self._use_recurrent_policy:
            x = x.squeeze(dim=1)
            x, rnn_states = self.actor.rnn(x, rnn_states, masks)
            x = x.reshape(-1, self.num_agents, self.hidden_size)
        agent_query = self.actor._agent_PE_Block(x)

        object_key = object_key.reshape(-1, self.num_agents, self.num_objects, self.hidden_size).permute(0, 2, 1, 3)
        object_key = self.actor._object_PMA(object_key.reshape(-1, self.num_agents, self.hidden_size))
        object_key = object_key.mean(dim = 1).reshape(-1, self.num_objects, self.hidden_size)
        
        output = self.actor._act_CAB(agent_query, object_key)
        
        action_log_probs, dist_entropy = self.actor.act_layer.evaluate_actions(
            output,
            action, 
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None
        )
        values, _ = self.critic(
            share_obs,
            rnn_states_critic,
            masks,
        )
        
        return values, action_log_probs, dist_entropy
    
    def act(self, share_obs, obs, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            
        actions, _, rnn_states = self.actor(
            obs,
            rnn_states,
            masks,
            available_actions,
            deterministic,
        )
        return actions, rnn_states 