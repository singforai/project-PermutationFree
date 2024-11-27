import torch
import torch.nn as nn 
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer

from algorithms.utils.mast_utils import SetAttentionBlock, PoolingMultiheadAttention, RFF
from algorithms.utils.set_utils import SAB, PMA

class Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, num_agents, num_objects, device):
        super(Actor, self).__init__()        
        
        self.num_agents: int = num_agents
        self.num_objects: int = num_objects 
        
        self._gain: float = args["gain"]
        self.num_head: int = args["n_head"]
        
        self.hidden_size: int = args["hidden_size"]
        self._recurrent_N: int = args["recurrent_N"]
        self.n_sab_blocks: int = args["n_sab_blocks"]
        self.num_seed_vector: int = args["n_seed_vector"]
        self._use_orthogonal: bool = args["use_orthogonal"]
        self.use_recurrent_policy: int = args["use_recurrent_policy"]
        
        self.input_dim = obs_space // self.num_objects

        self.base = nn.Sequential(
            nn.Linear(
                self.input_dim, self.hidden_size
            ),
            nn.ReLU(),
            nn.Linear(
                self.hidden_size, self.hidden_size
            ), 
        )
        # self._feature_PF_Block = nn.Sequential(
        #     SAB(
        #         hidden_size =  self.hidden_size,
        #         num_heads = self.num_head,
        #         sample_size = self.num_objects,
        #     ),
        #     PMA(
        #         hidden_size =  self.hidden_size,
        #         num_seed_vector = self.num_seed_vector,
        #         num_heads = self.num_head,
        #         v_norm_samples = self.num_objects,
        #     )
        # )
        self._feature_PF_Block = nn.Sequential(
            SetAttentionBlock(
                d =  self.hidden_size,
                h = self.num_head,
                rff = RFF(self.hidden_size),
            ),
            PoolingMultiheadAttention(
                d =  self.hidden_size,
                k = self.num_seed_vector,
                h = self.num_head,
                rff = RFF(self.hidden_size),
            )
        )
        
        # self._agent_PE_Block = nn.Sequential(
        #     SAB(
        #         hidden_size =  self.hidden_size,
        #         num_heads = self.num_head,
        #         sample_size = self.num_agents,
        #     ),
        # )
        self._agent_PE_Block = nn.Sequential(
            SetAttentionBlock(
                d =  self.hidden_size,
                h = self.num_head,
                rff = RFF(self.hidden_size),
            ),
        )

        self.act_layer = ACTLayer(
            action_space = action_space,
            inputs_dim = self.hidden_size,
            use_orthogonal = self._use_orthogonal,
            gain = self._gain
        )
        
        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        x = obs.reshape(obs.shape[0], self.num_objects, self.input_dim) # (B * N_A) x N_O x D
        x = self.base(x)
        x = self._feature_PF_Block(x)
        x = x.mean(dim = 1)
        x = x.reshape(-1 ,self.num_agents, self.hidden_size)
        x = self._agent_PE_Block(x)
        x = x.reshape(-1, self.hidden_size)
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
        
        self.input_dim = obs_space // self.num_objects
        
        self.base = nn.Sequential(
            nn.Linear(
                self.input_dim, self.hidden_size
            ),
            nn.ReLU(),
            nn.Linear(
                self.hidden_size, self.hidden_size
            ), 
        )
        
        # self._feature_PF_Block = nn.Sequential(
        #     SAB(
        #         hidden_size =  self.hidden_size,
        #         num_heads = self.num_head,
        #         sample_size = self.num_objects,
        #     ),
        #     PMA(
        #         hidden_size =  self.hidden_size,
        #         num_seed_vector = self.num_seed_vector,
        #         num_heads = self.num_head,
        #         v_norm_samples = self.num_objects,
        #     )
        # )

        self._feature_PF_Block = nn.Sequential(
            SetAttentionBlock(
                d =  self.hidden_size,
                h = self.num_head,
                rff = RFF(self.hidden_size),
            ),
            PoolingMultiheadAttention(
                d =  self.hidden_size,
                k = self.num_seed_vector,
                h = self.num_head,
                rff = RFF(self.hidden_size),
            )
        )

        # self._agent_PF_Block = nn.Sequential(
        #     SAB(
        #         hidden_size =  self.hidden_size,
        #         num_heads = self.num_head,
        #         sample_size = self.num_agents,
        #     ),
        #     PMA(
        #         hidden_size =  self.hidden_size,
        #         num_seed_vector = self.num_seed_vector,
        #         num_heads = self.num_head,
        #         v_norm_samples = self.num_agents,
        #     )
        # )

        self._agent_PF_Block = nn.Sequential(
            SetAttentionBlock(
                d =  self.hidden_size,
                h = self.num_head,
                rff = RFF(self.hidden_size),
            ),
            PoolingMultiheadAttention(
                d =  self.hidden_size,
                k = self.num_seed_vector,
                h = self.num_head,
                rff = RFF(self.hidden_size),
            )
        )

        self.v_net = nn.Sequential(
            nn.Linear(
                (self.hidden_size), self.hidden_size // 2
            ),
            nn.ReLU(),
            nn.Linear(
                self.hidden_size // 2, 1
            ),
        )
            
        self.to(device)

    def forward(self, obs):
        x = obs.reshape(obs.shape[0], self.num_objects, self.input_dim) # (B * N_A) x N_O x D
        x = self.base(x)
        x = self._feature_PF_Block(x)
        x = x.mean(dim = 1)
        x = x.reshape(-1 ,self.num_agents, self.hidden_size)
        x = self._agent_PF_Block(x)
        x = x.mean(dim = 1)
        x = self.v_net(x)
        values = x.unsqueeze(1).repeat(1, self.num_agents, 1).reshape(-1, 1)
        return values