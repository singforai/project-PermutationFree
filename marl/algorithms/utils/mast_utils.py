import torch
import torch.nn as nn

class Attention(nn.Module):
    """Scaled Dot-Product Attention."""
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)


    def forward(self, queries, keys, values):
        attention = torch.bmm(queries, keys.transpose(1, 2)) / self.temperature
        attention = self.softmax(attention) # it has shape [b, n, m]
        return torch.bmm(attention, values) # it has shape [b, n, d]
            

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

        self.project_queries = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace = True),
            nn.Linear(d, d),
        )
        self.project_keys = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace = True),
            nn.Linear(d, d),
        )
        self.project_values = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace = True),
            nn.Linear(d, d),
        )
        self.attention = Attention(temperature=p**0.5)

    def forward(self, queries, keys, values):
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
        
        queries = queries.view(b, n, h, p)
        keys = keys.view(b, m, h, p)
        values = values.view(b, m, h, p)
        
        queries = queries.permute(2, 0, 1, 3).contiguous().view(h * b, n, p)
        keys = keys.permute(2, 0, 1, 3).contiguous().view(h * b, m, p)
        values = values.permute(2, 0, 1, 3).contiguous().view(h * b, m, p)

        output = self.attention(queries, keys, values)  # shape [h * b, n, p]
        output = output.view(h, b, n, p)
        output = output.permute(1, 2, 0, 3).contiguous().view(b, n, d) # shape [b, n, d]

        return output
    
class MultiheadAttentionBlock(nn.Module):

    def __init__(self, d, h):
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
        #self.layer_norm2 = nn.LayerNorm(d)
    
    def forward(self, x, y):
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
        return self.layer_norm1(x + self.multihead(x, y, y))

class SetAttentionBlock(nn.Module):

    def __init__(self, d, h):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.mab(x, x)

class CrossattentionBlock(nn.Module):

    def __init__(self, d, h):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h)

    def forward(self, x, y):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.mab(x, y)

class PoolingMultiheadAttention(nn.Module):

    def __init__(self, d, k, h):
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
        self.mab = MultiheadAttentionBlock(d, h)
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
        return self.mab(s, z)


class ActBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        
        self.project_queries = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace = True),
            nn.Linear(d, d),
        )
        self.project_keys = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace = True),
            nn.Linear(d, d),
        )
        
        self.temperature = d**0.5
          
    def forward(self, queries, keys):
        b, n, d = queries.size()
        _, m, _ = keys.size()
    
        queries = self.project_queries(queries)  # shape [b, n, d]
        keys = self.project_keys(keys)  # shape [b, m, d]

        queries = queries.view(b, n, 1, d)
        keys = keys.view(b, m, 1, d)

        queries = queries.permute(2, 0, 1, 3).contiguous().view(b, n, d)
        keys = keys.permute(2, 0, 1, 3).contiguous().view(b, m, d)
        
        return (torch.bmm(queries, keys.transpose(1, 2)) / self.temperature).reshape(b * n, m)

class CrossAttention(nn.Module):
    
    def __init__(self, d, k):
        super().__init__()
        
        self.act_block = ActBlock(d)
        
        self.no_attack_act = nn.Parameter(torch.randn(1, k, d))
        

    def forward(self, agent_query, object_key):
        batch_size = object_key.size(0)
        seed_vectors = self.no_attack_act.repeat([batch_size, 1, 1])
        object_keys = torch.cat((seed_vectors, object_key), dim=1)
        return self.act_block(agent_query, object_keys)