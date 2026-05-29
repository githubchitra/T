import math
import torch
import torch.nn as nn
from attention import MultiHeadAttention

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding as described in "Attention Is All You Need".
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encodings in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # We step by 2 to compute sine and cosine in alternate dimensions.
        # Handle odd dimensions gracefully, though embed_dim is normally even.
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        # If d_model is odd, the cosine terms slice will have one less column
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
            
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input embeddings of shape [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: Positionally-encoded embeddings of shape [batch_size, seq_len, d_model]
        """
        # Add the sinusoidal positional encodings to the input embeddings
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) containing two linear layers.
    FFN(x) = Activation(x W_1 + b_1) W_2 + b_2
    """
    def __init__(self, embed_dim: int, d_ff: int, dropout: float = 0.0, activation: str = 'gelu'):
        super().__init__()
        self.w_1 = nn.Linear(embed_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        if activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}. Choose 'gelu' or 'relu'.")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: Projected output of shape [batch_size, seq_len, embed_dim]
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block using Pre-Layer Normalization (Pre-LN).
    """
    def __init__(self, embed_dim: int, num_heads: int, d_ff: int, dropout: float = 0.0, activation: str = 'gelu'):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim, d_ff, dropout, activation)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim]
            mask (torch.Tensor, optional): Self-attention padding mask of shape [batch_size, seq_len, seq_len]
            
        Returns:
            torch.Tensor: Output representations of shape [batch_size, seq_len, embed_dim]
            torch.Tensor: Self-attention weights from this block
        """
        # 1. Pre-LN Self-Attention Sublayer
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.self_attn(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_out)
        
        # 2. Pre-LN Feed-Forward Sublayer
        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)
        
        return x, attn_weights


class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block using Pre-Layer Normalization (Pre-LN).
    Contains Masked Self-Attention and Cross-Attention.
    """
    def __init__(self, embed_dim: int, num_heads: int, d_ff: int, dropout: float = 0.0, activation: str = 'gelu'):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim, d_ff, dropout, activation)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, enc_outputs: torch.Tensor = None, tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): Decoder input of shape [batch_size, seq_len_tgt, embed_dim]
            enc_outputs (torch.Tensor, optional): Encoder outputs of shape [batch_size, seq_len_src, embed_dim]
            tgt_mask (torch.Tensor, optional): Mask for decoder self-attention.
            memory_mask (torch.Tensor, optional): Mask for decoder-encoder cross-attention.
            
        Returns:
            torch.Tensor: Decoder block output of shape [batch_size, seq_len_tgt, embed_dim]
            torch.Tensor: Self-attention weights
            torch.Tensor: Cross-attention weights (None if cross-attention is not active)
        """
        # 1. Pre-LN Masked Self-Attention Sublayer
        norm_x = self.norm1(x)
        self_attn_out, self_attn_weights = self.self_attn(norm_x, norm_x, norm_x, tgt_mask)
        x = x + self.dropout(self_attn_out)
        
        cross_attn_weights = None
        # 2. Pre-LN Cross-Attention Sublayer (only active if encoder outputs are provided)
        if enc_outputs is not None:
            norm_x = self.norm2(x)
            cross_attn_out, cross_attn_weights = self.cross_attn(norm_x, enc_outputs, enc_outputs, memory_mask)
            x = x + self.dropout(cross_attn_out)
            
        # 3. Pre-LN Feed-Forward Sublayer
        norm_x = self.norm3(norm_x if enc_outputs is None else x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)
        
        return x, self_attn_weights, cross_attn_weights
