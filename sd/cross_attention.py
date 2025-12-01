import torch
from torch import nn
import math

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=False, out_proj_bias=False):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x: (Batch_Size, Seq_Len_Q, Dim_Q)
        # y: (Batch_Size, Seq_Len_KV, Dim_KV)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Apply the query projection to x
        q = self.q_proj(x)
        q = q.view(interim_shape).transpose(1, 2)

        # Apply the key and value projections to y
        k = self.cross_attn_k(y)
        v = self.cross_attn_v(y)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Compute attention weights
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = nn.functional.softmax(weight, dim=-1)

        # Apply attention to values
        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)

        # Apply output projection
        output = self.out_proj(output)

        return output
