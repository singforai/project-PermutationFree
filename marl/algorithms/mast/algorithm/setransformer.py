import torch
import torch.nn as nn 
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer

from algorithms.utils.mast_utils import *

from utils.util import get_shape_from_obs_space

class Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, num_agents, num_objects, device):
        super(Actor, self).__init__()        
        
        self.num_agents: int = num_agents
        self.num_objects: int = num_objects 
        
        self._gain: float = args["gain"]
        self.num_head: int = args["n_head"]

        self.hidden_size: int = args["hidden_size"]
        self._recurrent_N: int = args["recurrent_N"]
        self._use_orthogonal: bool = args["use_orthogonal"]
        self.use_recurrent_policy: int = args["use_recurrent_policy"]
        
        self.base = nn.Sequential(
            nn.Linear(
                obs_space ,self.hidden_size
            ),
        )
        
        if self.use_recurrent_policy:
            self.rnn_layer = RNNLayer(
                inputs_dim = self.hidden_size,
                outputs_dim = self.hidden_size,
                recurrent_N = self._recurrent_N,
                use_orthogonal = self._use_orthogonal 
            )
        self.SAB_block = SetAttentionBlock(
                d = self.hidden_size,
                h = self.num_head,
                rff = RFF(self.hidden_size),
        )

        self.CAB_block = CrossAttentionBlock(
            d = self.hidden_size,
            h = self.num_head,
            rff = RFF(self.hidden_size)
        )
        
        self.SAB_block_ally = SetAttentionBlock(
                d = self.hidden_size,
                h = self.num_head,
                rff = RFF(self.hidden_size),
        )
        
        self.act_layer = ACTLayer(
            action_space = action_space,
            inputs_dim = self.hidden_size,
            use_orthogonal = self._use_orthogonal,
            gain = self._gain
        )
        
        self.to(device)

    def forward(self, obs, rnn_states, visible_masking, masks, available_actions=None, deterministic=False):
        x = self.base(obs)
        
        x = x.reshape(-1, self.num_objects, self.hidden_size)[:, : self.num_agents, :].reshape(-1, self.hidden_size)    



        # x = self.SAB_block(x, visible_masking = visible_masking)

        # agents = x[:, : self.num_agents, :]
        # enemys = x[:, self.num_agents :, :]
        # x = self.CAB_block(x = agents, y = enemys, visible_masking = visible_masking[:, :self.num_agents, self.num_agents:])
        
        # x = self.SAB_block_ally(x, visible_masking = visible_masking[:, :self.num_agents, :self.num_agents])
        
        # x = x.reshape(-1, self.hidden_size)
        actions, action_log_probs = self.act_layer(
            x,
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
        
        self.base = nn.Sequential(
            nn.Linear(
                obs_space ,self.hidden_size
            ),
        )

        self.CAB_block = CrossAttentionBlock(
            d = self.hidden_size,
            h = self.num_head,
            rff = RFF(self.hidden_size)
        )
        
        self.SAB_block_ally = SetAttentionBlock(
                d = self.hidden_size,
                h = self.num_head,
                rff = RFF(self.hidden_size),
        )
        
        self.PMA_block = PoolingMultiheadAttention(
            d = self.hidden_size,
            k = self.num_seed_vector, 
            h = self.num_head, 
            rff = RFF(self.hidden_size)

        )
        self.v_net = nn.Sequential(
            nn.Linear(
                (self.hidden_size * self.num_seed_vector), self.hidden_size // 2
            ),
            nn.ReLU(),
            nn.Linear(
                self.hidden_size // 2, self.num_agents
            ),
        )
            
        self.to(device)

    def forward(self, obs, visible_masking = None):
        x = self.base(obs)
        
        x = x.reshape(-1, self.num_objects, self.hidden_size)
        
        agents = x[:, : self.num_agents, :]
        enemys = x[:, self.num_agents:, :]
        
        x = self.CAB_block(x = agents, y = enemys)
        
        x = self.SAB_block_ally(x)
        
        x = self.PMA_block(x)
        x = x.reshape(-1, self.num_seed_vector * self.hidden_size)
        x = self.v_net(x)
        values = x.reshape(-1, 1)
        return values