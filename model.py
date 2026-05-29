import torch
import torch.nn as nn
from transformer_block import PositionalEncoding, EncoderBlock, DecoderBlock

def init_weights(module: nn.Module):
    """
    Initializes module weights:
    - Xavier uniform for all Linear layers
    - Normal distribution (mean=0.0, std=0.02) for Embedding layers
    - Gamma = 1.0, Beta = 0.0 for LayerNorm layers
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder consisting of a stack of EncoderBlocks, 
    an Embedding layer, and Positional Encoding.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): Input tokens of shape [batch_size, seq_len]
            mask (torch.Tensor, optional): Input mask of shape [batch_size, seq_len, seq_len]
            
        Returns:
            torch.Tensor: Encoded features of shape [batch_size, seq_len, embed_dim]
            list: Attention weight matrices for each encoder layer
        """
        out = self.embedding(x)
        out = self.pos_encoding(out)
        
        all_attention_weights = []
        for layer in self.layers:
            out, attn_weights = layer(out, mask)
            all_attention_weights.append(attn_weights)
            
        out = self.norm(out)
        return out, all_attention_weights


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder consisting of a stack of DecoderBlocks, 
    an Embedding layer, and Positional Encoding.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, enc_outputs: torch.Tensor = None, tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): Decoder input tokens of shape [batch_size, seq_len_tgt]
            enc_outputs (torch.Tensor, optional): Encoder outputs of shape [batch_size, seq_len_src, embed_dim]
            tgt_mask (torch.Tensor, optional): Target self-attention mask
            memory_mask (torch.Tensor, optional): Source-target cross-attention mask
            
        Returns:
            torch.Tensor: Decoded features of shape [batch_size, seq_len_tgt, embed_dim]
            list: Self-attention weights for each decoder layer
            list: Cross-attention weights for each decoder layer
        """
        out = self.embedding(x)
        out = self.pos_encoding(out)
        
        all_self_attn_weights = []
        all_cross_attn_weights = []
        for layer in self.layers:
            out, self_attn, cross_attn = layer(out, enc_outputs, tgt_mask, memory_mask)
            all_self_attn_weights.append(self_attn)
            all_cross_attn_weights.append(cross_attn)
            
        out = self.norm(out)
        return out, all_self_attn_weights, all_cross_attn_weights


class TransformerSeq2Seq(nn.Module):
    """
    A full Seq2Seq Transformer model combining the Encoder and Decoder.
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, embed_dim, num_layers, num_heads, d_ff, max_len, dropout, activation)
        self.decoder = TransformerDecoder(tgt_vocab_size, embed_dim, num_layers, num_heads, d_ff, max_len, dropout, activation)
        self.fc_out = nn.Linear(embed_dim, tgt_vocab_size)
        
        # Apply weight initialization
        self.apply(init_weights)
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None):
        """
        Args:
            src (torch.Tensor): Source tokens of shape [batch_size, seq_len_src]
            tgt (torch.Tensor): Target tokens of shape [batch_size, seq_len_tgt]
            src_mask (torch.Tensor, optional): Encoder mask
            tgt_mask (torch.Tensor, optional): Decoder self-attention mask
            memory_mask (torch.Tensor, optional): Cross-attention mask
            
        Returns:
            torch.Tensor: Logits over vocabulary of shape [batch_size, seq_len_tgt, tgt_vocab_size]
            dict: Attention weight collections
        """
        enc_outputs, enc_attn = self.encoder(src, src_mask)
        dec_outputs, dec_self_attn, dec_cross_attn = self.decoder(tgt, enc_outputs, tgt_mask, memory_mask)
        logits = self.fc_out(dec_outputs)
        
        return logits, {
            "encoder_attention": enc_attn,
            "decoder_self_attention": dec_self_attn,
            "decoder_cross_attention": dec_cross_attn
        }


class TransformerLM(nn.Module):
    """
    Decoder-only Transformer Language Model (like GPT).
    Constructed by stacking EncoderBlocks and applying a causal mask to self-attention
    to prevent future-token information leakage.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
        # Apply custom initialization
        self.apply(init_weights)
        
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generates a lower-triangular mask of shape [seq_len, seq_len].
        """
        return torch.tril(torch.ones(seq_len, seq_len, device=device))
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): Input tokens of shape [batch_size, seq_len]
            mask (torch.Tensor, optional): Optional padding/user mask of shape [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Logits of shape [batch_size, seq_len, vocab_size]
            list: Attention weight matrices from each layer
        """
        seq_len = x.size(1)
        causal_mask = self._generate_causal_mask(seq_len, x.device)  # [seq_len, seq_len]
        
        # Combine causal mask with external mask (e.g. padding mask) if provided
        if mask is not None:
            # mask: [batch_size, seq_len] -> [batch_size, 1, seq_len] -> [batch_size, seq_len, seq_len]
            mask = mask.unsqueeze(1) & causal_mask.unsqueeze(0)
        else:
            mask = causal_mask
            
        out = self.embedding(x)
        out = self.pos_encoding(out)
        
        all_attn_weights = []
        for layer in self.layers:
            out, attn_weights = layer(out, mask)
            all_attn_weights.append(attn_weights)
            
        out = self.norm(out)
        logits = self.fc_out(out)
        
        return logits, all_attn_weights


class LSTMCell(nn.Module):
    """
    A standard LSTM cell implemented completely from scratch.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Combine input-to-hidden and hidden-to-hidden projections into one Linear layer
        self.xh_proj = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input at time step t of shape [batch_size, input_dim]
            h_prev (torch.Tensor): Hidden state at step t-1 of shape [batch_size, hidden_dim]
            c_prev (torch.Tensor): Cell state at step t-1 of shape [batch_size, hidden_dim]
            
        Returns:
            torch.Tensor: Updated hidden state h_t of shape [batch_size, hidden_dim]
            torch.Tensor: Updated cell state c_t of shape [batch_size, hidden_dim]
        """
        combined = torch.cat([x, h_prev], dim=-1)  # [batch_size, input_dim + hidden_dim]
        gates = self.xh_proj(combined)             # [batch_size, 4 * hidden_dim]
        
        # Split gates: Forget, Input, Candidate Cell, Output
        f_gate, i_gate, g_gate, o_gate = gates.chunk(4, dim=-1)
        
        f_gate = torch.sigmoid(f_gate)
        i_gate = torch.sigmoid(i_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)
        
        c_next = f_gate * c_prev + i_gate * g_gate
        h_next = o_gate * torch.tanh(c_next)
        
        return h_next, c_next


class BaselineLSTM(nn.Module):
    """
    A multi-layer LSTM language model implemented from scratch.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            LSTMCell(embed_dim if l == 0 else hidden_dim, hidden_dim)
            for l in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.apply(init_weights)
        
    def forward(self, x: torch.Tensor, states=None):
        """
        Args:
            x (torch.Tensor): Input tokens of shape [batch_size, seq_len]
            states (list of tuples, optional): Previous hidden/cell states for each layer.
            
        Returns:
            torch.Tensor: Logits of shape [batch_size, seq_len, vocab_size]
            list: Updated hidden/cell states for each layer
        """
        batch_size, seq_len = x.size()
        device = x.device
        
        if states is None:
            states = [
                (torch.zeros(batch_size, self.hidden_dim, device=device),
                 torch.zeros(batch_size, self.hidden_dim, device=device))
                for _ in range(self.num_layers)
            ]
            
        embeds = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        outputs = []
        for t in range(seq_len):
            x_t = embeds[:, t, :]  # Input at time step t: [batch_size, embed_dim]
            
            new_states = []
            for layer_idx, cell in enumerate(self.cells):
                h_prev, c_prev = states[layer_idx]
                h_next, c_next = cell(x_t, h_prev, c_prev)
                new_states.append((h_next, c_next))
                x_t = h_next  # The hidden state is the input to the next layer
                
            states = new_states
            outputs.append(x_t.unsqueeze(1))
            
        outputs = torch.cat(outputs, dim=1)  # [batch_size, seq_len, hidden_dim]
        logits = self.fc_out(outputs)
        
        return logits, states
