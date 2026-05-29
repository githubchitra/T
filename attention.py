import math
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.
    Computes: softmax(Q K^T / sqrt(d_k) + Mask) V
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (torch.Tensor): Queries of shape [batch_size, num_heads, seq_len_q, d_k]
            k (torch.Tensor): Keys of shape [batch_size, num_heads, seq_len_k, d_k]
            v (torch.Tensor): Values of shape [batch_size, num_heads, seq_len_v, d_v] (d_k == d_v)
            mask (torch.Tensor, optional): Mask tensor of shape broadcastable to [batch_size, num_heads, seq_len_q, seq_len_k].
                                         True/1 means keep, False/0 means mask out.
                                         
        Returns:
            torch.Tensor: Weighted context values of shape [batch_size, num_heads, seq_len_q, d_v]
            torch.Tensor: Attention weights of shape [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        d_k = q.size(-1)
        # Compute raw similarity scores
        # Shape: [batch_size, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # Mask contains False (0) for positions to ignore. Fill with a large negative value
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # Compute probabilities
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights (standard Transformer behavior)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values
        # Shape: [batch_size, num_heads, seq_len_q, d_v]
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention from scratch.
    Splits the embedding dimension across multiple attention heads.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, f"Embedding dimension {embed_dim} must be divisible by number of heads {num_heads}."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        
        # Projections to Query, Key, and Value spaces
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Project concatenated head outputs back to embedding dimension
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.attention = ScaledDotProductAttention(dropout)
        
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (torch.Tensor): Queries of shape [batch_size, seq_len_q, embed_dim]
            k (torch.Tensor): Keys of shape [batch_size, seq_len_k, embed_dim]
            v (torch.Tensor): Values of shape [batch_size, seq_len_v, embed_dim]
            mask (torch.Tensor, optional): Mask of shape [batch_size, seq_len_q, seq_len_k]
                                         or broadcastable shape.
                                         
        Returns:
            torch.Tensor: Attention context outputs of shape [batch_size, seq_len_q, embed_dim]
            torch.Tensor: Attention weights of shape [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = q.size(0)
        
        # 1. Project inputs to Q, K, V and reshape to split heads
        # Shape: [batch_size, num_heads, seq_len, d_k]
        q_proj = self.q_proj(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k_proj = self.k_proj(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v_proj = self.v_proj(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Adjust mask dimension for multi-head attention if needed
        # If mask is [batch_size, seq_len_q, seq_len_k], we shape to [batch_size, 1, seq_len_q, seq_len_k]
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)
            
        # 3. Apply Scaled Dot-Product Attention on projected representations
        # context shape: [batch_size, num_heads, seq_len_q, d_k]
        # attn_weights shape: [batch_size, num_heads, seq_len_q, seq_len_k]
        context, attn_weights = self.attention(q_proj, k_proj, v_proj, mask)
        
        # 4. Concatenate heads back together
        # Shape: [batch_size, seq_len_q, num_heads * d_k] -> [batch_size, seq_len_q, embed_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # 5. Apply final linear projection
        output = self.out_proj(context)
        
        return output, attn_weights
