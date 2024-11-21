import torch
import torch.nn as nn


# class Token_Mixing(nn.Module) : 
#     def __init__(self, input) : 
#         super(Token_Mixing, self).__init__()

#         self.Layer_Norm = nn.LayerNorm(input[-2]) 
#         self.MLP = nn.Sequential(
#             nn.Linear(input[-2], input[-2]),
#             nn.GELU(),
#             nn.Linear(input[-2], input[-2])
#         )

#     def forward(self, x) : # (B * N_A) x N_O x H
#         output = self.Layer_Norm(x.transpose(2,1))  # (B * N_A) x H x N_O
#         output = self.MLP(output) # (B * N_A) x H x N_O

#         output = output.transpose(2,1) # (B * N_A) x N_O x H

#         return output + x # (B * N_A) x N_O x H

# class Channel_Mixing(nn.Module) :
#     def __init__(self, input) : # 
#         super(Channel_Mixing, self).__init__()

#         self.Layer_Norm = nn.LayerNorm(input[-1])
#         self.MLP = nn.Sequential(
#             nn.Linear(input[-1], input[-1]),
#             nn.GELU(),
#             nn.Linear(input[-1], input[-1])
#         )
    
#     def forward(self, x) : # (B * N_A) x N_O x H
#         output = self.Layer_Norm(x) # (B * N_A) x N_O x H
#         output = self.MLP(output)  # (B * N_A) x N_O x H
#         return output + x  # (B * N_A) x N_O x H

# class Mixer_Layer(nn.Module) :
#     def __init__(self, input) : # 
#         super(Mixer_Layer, self).__init__()

#         self.mixer_layer = nn.Sequential(
#             Token_Mixing(input),
#             Channel_Mixing(input)
#         )
#     def forward(self, x) :
#         return self.mixer_layer(x)
    

# class MLP_Mixer(nn.Module):
#     def __init__(self, num_agents, num_objects, mixer_hidden_size, input_dim, n_block = 3):
#         super(MLP_Mixer, self).__init__()
#         self.num_agents = num_agents
#         self.num_objects = num_objects
#         self.mixer_hidden_size = mixer_hidden_size
#         self.input_dim = input_dim
        
#         self.base = nn.Sequential(
#             nn.Linear(
#                 self.input_dim, self.mixer_hidden_size
#             ),
#             nn.GELU(),
#             nn.Linear(
#                 self.mixer_hidden_size, self.mixer_hidden_size
#             ), 
#         )
        
#         self.mlp_mixer = nn.Sequential()
        
#         for i in range(n_block) : # Mixer Layer를 N번 쌓아준다
#             self.mlp_mixer.add_module("Mixer_Layer_" + str(i), Mixer_Layer((self.num_objects, self.mixer_hidden_size)))
            
#         self.global_average_Pooling = nn.LayerNorm([self.num_objects, self.mixer_hidden_size])
            
#     def forward(self, obs):
#         x = obs.reshape(obs.shape[0] * obs.shape[1], self.num_objects, self.input_dim) # (B * N_A) x N_O x D
#         x = self.base(x) # (B * N_A) x N_O x H
#         x = self.mlp_mixer(x) # (B * N_A) x N_O x H
#         x = self.global_average_Pooling(x) # (B * N_A) x N_O x H
#         x = x.mean(1) # (B * N_A) x H
#         x = x.reshape(obs.shape[0], self.num_agents, self.mixer_hidden_size) # B x N_A x H
#         return x





class Attention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, temperature):
        super().__init__()

        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, queries, keys, values, visible_masking):
        """
        It is equivariant to permutations
        of the batch dimension (`b`).

        It is equivariant to permutations of the
        second dimension of the queries (`n`).

        It is invariant to permutations of the
        second dimension of keys and values (`m`).

        Arguments:
            queries: a float tensor with shape [b, n, d].
            keys: a float tensor with shape [b, m, d].
            values: a float tensor with shape [b, m, d'].
        Returns:
            a float tensor with shape [b, n, d'].
        """

        attention = torch.bmm(queries, keys.transpose(1, 2))
        attention = attention / self.temperature
        
        if visible_masking is not None:
            attention = attention.masked_fill(visible_masking == 0.0, float(-1e10))

        attention = self.softmax(attention)
        # it has shape [b, n, m]
        return torch.bmm(attention, values)


class MultiheadAttention(nn.Module):
    def __init__(self, d, h):
        """
        Arguments:
            d: an integer, dimension of queries and values.
                It is assumed that input and
                output dimensions are the same.
            h: an integer, number of heads.
        """
        super().__init__()

        assert d % h == 0
        self.h = h

        # everything is projected to this dimension
        p = d // h

        self.project_queries = nn.Linear(d, d)
        self.project_keys = nn.Linear(d, d)
        self.project_values = nn.Linear(d, d)
        self.concatenation = nn.Linear(d, d)
        self.attention = Attention(temperature=p**0.5)

    def forward(self, queries, keys, values, visible_masking):
        """
        Arguments:
            queries: a float tensor with shape [b, n, d].
            keys: a float tensor with shape [b, m, d].
            values: a float tensor with shape [b, m, d].
        Returns:
            a float tensor with shape [b, n, d].
        """

        h = self.h
        b, n, d = queries.size()
        _, m, _ = keys.size()
        p = d // h
        
        queries = self.project_queries(queries)  # shape [b, n, d]
        keys = self.project_keys(keys)  # shape [b, m, d]
        values = self.project_values(values)  # shape [b, m, d]
        visible_masking = visible_masking # shape [b, n, m]
        
        queries = queries.view(b, n, h, p)
        keys = keys.view(b, m, h, p)
        values = values.view(b, m, h, p)
        if visible_masking is not None:
            visible_masking = visible_masking.unsqueeze(3).repeat(1, 1, 1, h) # shape [b, n, m, h]
        
        queries = queries.permute(2, 0, 1, 3).contiguous().view(h * b, n, p)
        keys = keys.permute(2, 0, 1, 3).contiguous().view(h * b, m, p)
        values = values.permute(2, 0, 1, 3).contiguous().view(h * b, m, p)
        if visible_masking is not None:
            visible_masking = visible_masking.permute(3, 0, 1, 2).contiguous().view(h * b, n, m)

        output = self.attention(queries, keys, values, visible_masking)  # shape [h * b, n, p]
        output = output.view(h, b, n, p)
        output = output.permute(1, 2, 0, 3).contiguous().view(b, n, d)
        output = self.concatenation(output)  # shape [b, n, d]

        return output
    
class MultiheadAttentionBlock(nn.Module):

    def __init__(self, d, h, rff):
        """
        Arguments:
            d: an integer, input dimension.
            h: an integer, number of heads.
            rff: a module, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super().__init__()

        self.multihead = MultiheadAttention(d, h)
        self.layer_norm1 = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.rff = rff

    def forward(self, x, y, visible_masking):
        """
        It is equivariant to permutations of the
        second dimension of tensor x (`n`).

        It is invariant to permutations of the
        second dimension of tensor y (`m`).

        Arguments:
            x: a float tensor with shape [b, n, d].
            y: a float tensor with shape [b, m, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        h = self.layer_norm1(x + self.multihead(x, y, y, visible_masking))
        return self.layer_norm2(h + self.rff(h))

class SetAttentionBlock(nn.Module):

    def __init__(self, d, h, rff):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff)

    def forward(self, x, visible_masking = None):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.mab(x, x, visible_masking)
    
class CrossAttentionBlock(nn.Module):

    def __init__(self, d, h, rff):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff)

    def forward(self, x, y, visible_masking = None):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
            y: a float tensor with shape [b, m, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.mab(x, y, visible_masking)


class PoolingMultiheadAttention(nn.Module):

    def __init__(self, d, k, h, rff):
        """
        Arguments:
            d: an integer, input dimension.
            k: an integer, number of seed vectors.
            h: an integer, number of heads.
            rff: a module, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff)
        self.seed_vectors = nn.Parameter(torch.randn(1, k, d))

    def forward(self, z):
        """
        Arguments:
            z: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, k, d].
        """
        b = z.size(0)
        s = self.seed_vectors
        s = s.repeat([b, 1, 1])  # shape [b, k, d]

        # note that in the original paper
        # they return mab(s, rff(z))
        return self.mab(s, z, visible_masking = None)

class RFF(nn.Module):
    """
    Row-wise FeedForward layers.
    """
    def __init__(self, d):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.layers(x)