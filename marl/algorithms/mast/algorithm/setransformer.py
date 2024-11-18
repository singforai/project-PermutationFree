import torch
import torch.nn as nn 
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer

from algorithms.utils.mast_utils import *

from utils.util import get_shape_from_obs_space

class Encoder(nn.Module):
    def __init__(self, args, obs_space, action_space, num_agents, num_objects, device):
        super(Encoder, self).__init__()        
        
        self.num_agents: int = num_agents
        self.num_objects: int = num_objects 
        
        self._gain: float = args["gain"]
        self.num_head: int = args["n_head"]
        self.obs_space: int = obs_space
        self.hidden_size: int = args["hidden_size"]
        self._recurrent_N: int = args["recurrent_N"]
        self._use_orthogonal: bool = args["use_orthogonal"]
        self.use_recurrent_policy: int = args["use_recurrent_policy"]
        
        self.base = nn.Sequential(
            nn.Linear(
                self.obs_space ,self.hidden_size
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
        if self.use_recurrent_policy:
            x, rnn_states = self.rnn_layer(
                x, 
                rnn_states, 
                masks
            )
        x = x.reshape(-1, self.num_objects, self.hidden_size)

        x = self.SAB_block(x, visible_masking = visible_masking)

        agents = x[:, : self.num_agents, :]
        enemys = x[:, self.num_agents :, :]
        x = self.CAB_block(x = agents, y = enemys, visible_masking = visible_masking[:, :self.num_agents, self.num_agents:])
        
        z = self.SAB_block_ally(x, visible_masking = visible_masking[:, :self.num_agents, :self.num_agents])
        
        x = x.reshape(-1, self.hidden_size)
        actions, action_log_probs = self.act_layer(
            x,
            available_actions,
            deterministic
        )
        return actions, action_log_probs, rnn_states, z

class Decoder(nn.Module):
    def __init__(self, args, num_agents, num_objects, device):
        super(Decoder, self).__init__()
    
        self.num_agents: int = num_agents
        self.num_objects: int = num_objects 
    
        self.num_head: int = args["n_head"]
        self.hidden_size: int = args["hidden_size"]
        self._recurrent_N: int = args["recurrent_N"]
        self.num_seed_vector: int = args["n_seed_vector"]
        self._use_orthogonal: bool = args["use_orthogonal"]
        
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
                self.hidden_size // 2, 1
            ),
        )
            
        self.to(device)

    def forward(self, z):
        x = self.PMA_block(z)
        x = x.reshape(-1, self.num_seed_vector * self.hidden_size)
        x = self.v_net(x)
        values = x.unsqueeze(1).repeat(1, self.num_agents, 1).reshape(-1, 1)
        return values


class MultiAgentSetTransformer(nn.Module):
    def __init__(self, args, obs_space, action_space, num_agents, num_objects, device=torch.device("cpu")):
        super(MultiAgentSetTransformer, self).__init__()
        
        self.num_agents: int = num_agents
        self.num_objects: int = num_objects 
        
        self.action_space = action_space
        
        self.hidden_size = args["hidden_size"]
        self._use_recurrent_policy = args["use_recurrent_policy"]
        self._use_policy_active_masks = args["use_policy_active_masks"]

        self.encoder = Encoder(
            args = args,
            obs_space = obs_space,
            action_space = self.action_space,
            num_agents = num_agents,
            num_objects = self.num_objects,
            device = device
            
        )
        self.decoder = Decoder(
            args = args,
            num_agents = num_agents,
            num_objects = self.num_objects,
            device = device
        )
        self.to(device)

    def evaluate_actions(self, obs, rnn_states, visible_masking, action, masks, available_actions=None, active_masks=None):
        x = self.encoder.base(obs)
        if self._use_recurrent_policy:
            x, _ = self.encoder.rnn_layer(
                x, 
                rnn_states, 
                masks
            )
        x = x.reshape(-1, self.num_objects, self.hidden_size)
        x = self.encoder.SAB_block(x, visible_masking = visible_masking)

        agents = x[:, :self.num_agents, :]
        enemys = x[:, self.num_agents:, :]
        
        x = self.encoder.CAB_block(x = agents, y = enemys, visible_masking = visible_masking[:, :self.num_agents, self.num_agents:])
        
        z = self.encoder.SAB_block_ally(x, visible_masking = visible_masking[:, :self.num_agents, :self.num_agents])
        
        x = x.reshape(-1, self.hidden_size)
        action_log_probs, dist_entropy = self.encoder.act_layer.evaluate_actions(
            x,
            action,
            available_actions, 
            active_masks = active_masks if self._use_policy_active_masks else None
        )
        values = self.decoder(z)
        return values, action_log_probs, dist_entropy
