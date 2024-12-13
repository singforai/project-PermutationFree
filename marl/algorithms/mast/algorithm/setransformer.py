import torch
import torch.nn as nn 
from algorithms.utils.mlp import MLPBase   
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ActLayer

from algorithms.utils.mast_utils import SetAttentionBlock, PoolingMultiheadAttention, CrossAttention, CrossattentionBlock

class Actor(nn.Module):
    def __init__(self, args, obs_space, num_agents, num_objects, n_actions_no_attack, device):
        super(Actor, self).__init__()        
        
        self.num_agents: int = num_agents
        self.num_objects: int = num_objects 
        self.n_actions_no_attack: int = n_actions_no_attack
        
        self._gain: float = args["gain"]
        self.num_head: int = args["n_head"]
        
        self.hidden_size: int = args["hidden_size"]
        self._recurrent_N: int = args["recurrent_N"]
        self.num_seed_vector: int = args["n_seed_vector"]
        self._use_orthogonal: bool = args["use_orthogonal"]
        self._use_recurrent_policy: int = args["use_recurrent_policy"]
        
        self.input_dim = obs_space // self.num_objects
        
        
        self.base = MLPBase(args, self.input_dim)
        
        if self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self._feature_SAB = SetAttentionBlock(
            d =  self.hidden_size,
            h = self.num_head,
        )
        self._feature_PMA = PoolingMultiheadAttention(
            d =  self.hidden_size,
            k = self.num_seed_vector,
            h = self.num_head,
        )

        self._object_PMA = PoolingMultiheadAttention(
            d =  self.hidden_size,
            k = self.num_seed_vector,
            h = self.num_head,
        )

        self._agent_PE_Block = SetAttentionBlock(
            d =  self.hidden_size,
            h = self.num_head,
        )
        
        self._act_CAB = CrossAttention(
            d =  self.hidden_size,
            k = self.n_actions_no_attack
        )
        
        self.act_layer = ActLayer()

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        
        
        x = obs.reshape(obs.shape[0], self.num_objects, self.input_dim) # (B * N_A) x N_O x D
        x = self.base(x)
        object_key = self._feature_SAB(x)
        x = self._feature_PMA(object_key)
        x = x.mean(dim = 1)
        if self._use_recurrent_policy:
            x = x.squeeze(dim=1)
            x, rnn_states = self.rnn(x, rnn_states, masks)
            x = x.reshape(-1 ,self.num_agents, self.hidden_size)
        agent_query = self._agent_PE_Block(x)
     
        object_key = object_key.reshape(-1, self.num_agents, self.num_objects, self.hidden_size).permute(0, 2, 1, 3)
        object_key = self._object_PMA(object_key.reshape(-1, self.num_agents, self.hidden_size))
        object_key = object_key.mean(dim = 1).reshape(-1, self.num_objects, self.hidden_size)
        
        output = self._act_CAB(agent_query, object_key)
        actions, action_log_probs = self.act_layer(
            output,
            available_actions,
            deterministic
        )

        return actions, action_log_probs, rnn_states

class Critic(nn.Module):
    def __init__(self, args, obs_space, num_agents, num_objects, device):
        super(Critic, self).__init__()
    
        self.num_agents: int = num_agents
        self.num_objects: int = num_objects 
    
        self.num_head: int = args["n_head"]
        self.hidden_size: int = args["hidden_size"]
        self._recurrent_N: int = args["recurrent_N"]
        self.num_seed_vector: int = args["n_seed_vector"]
        self._use_orthogonal: bool = args["use_orthogonal"]
        self._use_recurrent_policy: int = args["use_recurrent_policy"]
        
        self.input_dim = obs_space // self.num_objects
        
        self.base = MLPBase(args, self.input_dim)

        if self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self._feature_PF_Block = nn.Sequential(
            SetAttentionBlock(
                d =  self.hidden_size,
                h = self.num_head,
            ),
            PoolingMultiheadAttention(
                d =  self.hidden_size,
                k = self.num_seed_vector,
                h = self.num_head,
            )
        )
        self._agent_PI_Block = nn.Sequential(
            PoolingMultiheadAttention(
                d =  self.hidden_size,
                k = self.num_seed_vector,
                h = self.num_head,
            )
        )

        self.v_net = nn.Linear(self.hidden_size, 1)
            
        self.to(device)

    def forward(self, share_obs, rnn_states_critic, masks):
        x = share_obs.reshape(share_obs.shape[0], self.num_objects, self.input_dim) # (B * N_A) x N_O x D
        x = self.base(x)
        x = self._feature_PF_Block(x)
        x = x.mean(dim = 1)
        if self._use_recurrent_policy:
            x = x.squeeze(dim=1)
            x, rnn_states_critic = self.rnn(x, rnn_states_critic, masks)
        x = x.reshape(-1 ,self.num_agents, self.hidden_size)
        x = self._agent_PI_Block(x)
        x = x.mean(dim = 1)
        x = self.v_net(x)
        values = x.unsqueeze(1).repeat(1, self.num_agents, 1).reshape(-1, 1)
        return values, rnn_states_critic
        
        