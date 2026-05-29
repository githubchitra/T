import os
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import TransformerLM, BaselineLSTM

class CharDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for character-level training data.
    """
    def __init__(self, text: str, seq_len: int, chars: list = None):
        self.seq_len = seq_len
        if chars is not None:
            self.chars = chars
        else:
            self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        # Filter text to ensure all characters are in the vocabulary when resuming
        self.data = [self.char_to_ix[ch] for ch in text if ch in self.char_to_ix]
        
    def __len__(self):
        # Calculate number of non-overlapping chunks of length seq_len
        return (len(self.data) - 1) // self.seq_len
        
    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        chunk = self.data[start_idx : start_idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class NoamScheduler:
    """
    Learning rate scheduler as defined in the "Attention Is All You Need" paper.
    lr = factor * (embed_dim ** -0.5) * min(step_num ** -0.5, step_num * warmup_steps ** -1.5)
    """
    def __init__(self, optimizer: torch.optim.Optimizer, embed_dim: int, warmup_steps: int = 4000, factor: float = 1.0):
        self.optimizer = optimizer
        self.embed_dim = embed_dim
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0
        # Initialize learning rate for step 1 immediately
        self.step()
        
    def step(self) -> float:
        """
        Updates the learning rate in the optimizer based on the current step.
        """
        self.step_num += 1
        lr = self.factor * (self.embed_dim ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
        
    def get_lr(self) -> float:
        """
        Returns the current learning rate.
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


def generate_text(model, prompt, vocab, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.2, device='cpu'):
    """
    Generates text autoregressively using the trained model with advanced sampling controls:
    - Temperature scaling
    - Top-k filtering (k=50)
    - Top-p (nucleus) filtering (p=0.9)
    - Repetition penalty (penalty factor 1.2)
    - Ensures a space is added after the prompt if not already present
    """
    model.eval()
    
    # Reconstruct token mappings from vocab list
    char_to_ix = {ch: i for i, ch in enumerate(vocab)}
    ix_to_char = {i: ch for i, ch in enumerate(vocab)}
    
    # Retrieve model's context sequence length (default to 128)
    seq_len = getattr(model, 'seq_len', 128)
    
    # Ensure space after the prompt if not already present
    if prompt and not prompt.endswith(" "):
        prompt = prompt + " "
        
    current_seq = [char_to_ix.get(ch, 0) for ch in prompt]
    
    if not current_seq:
        current_seq = [0]
        
    generated = list(current_seq)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop inputs to fit context length limit
            input_seq = generated[-seq_len:]
            x = torch.tensor([input_seq], dtype=torch.long, device=device)
            
            if isinstance(model, TransformerLM):
                logits, _ = model(x)
                last_logit = logits[0, -1, :].clone()
            elif isinstance(model, BaselineLSTM):
                logits, _ = model(x)
                last_logit = logits[0, -1, :].clone()
            else:
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                last_logit = out[0, -1, :].clone()
                
            # 1. Apply repetition penalty
            if repetition_penalty != 1.0 and len(generated) > 0:
                for token in set(generated):
                    if last_logit[token] > 0:
                        last_logit[token] /= repetition_penalty
                    else:
                        last_logit[token] *= repetition_penalty
                        
            # 2. Apply temperature scaling
            if temperature > 0.0:
                last_logit = last_logit / temperature
                
            # 3. Apply top-k filtering
            if top_k > 0:
                v, _ = torch.topk(last_logit, min(top_k, last_logit.size(-1)))
                threshold = v[-1]
                last_logit[last_logit < threshold] = float('-inf')
                
            # 4. Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(last_logit, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift indices to keep the first token exceeding top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                last_logit[indices_to_remove] = float('-inf')
                
            # 5. Sampling
            if temperature <= 0.0:
                next_idx = torch.argmax(last_logit).item()
            else:
                probs = torch.softmax(last_logit, dim=-1)
                if torch.isnan(probs).any() or probs.sum() == 0:
                    next_idx = torch.argmax(last_logit).item()
                else:
                    next_idx = torch.multinomial(probs, 1).item()
                    
            generated.append(next_idx)
            
    return "".join([ix_to_char[idx] for idx in generated])


def plot_attention_heatmap(attn_weights: list, text: str, layer_idx: int, head_idx: int, save_path: str):
    """
    Plots attention weight heatmaps for a given input sequence.
    
    Args:
        attn_weights (list): A list of attention weight tensors from the model layers.
                             Each tensor has shape [batch_size, num_heads, seq_len, seq_len].
        text (str): The corresponding input text sequence.
        layer_idx (int): 0-indexed layer index to plot.
        head_idx (int): 0-indexed head index to plot.
        save_path (str): File path where the plot image should be saved.
    """
    if layer_idx >= len(attn_weights):
        raise ValueError(f"layer_idx {layer_idx} exceeds the number of layers ({len(attn_weights)}).")
        
    # Get shape: [batch, heads, seq, seq]
    layer_weights = attn_weights[layer_idx]
    num_heads = layer_weights.size(1)
    
    if head_idx >= num_heads:
        raise ValueError(f"head_idx {head_idx} exceeds the number of attention heads ({num_heads}).")
        
    # Extract the single batch and head slice
    grid_weights = layer_weights[0, head_idx].cpu().detach().numpy()
    
    tokens = list(text)
    seq_len = len(tokens)
    
    # Crop to the actual sequence length if the padded tensor is larger
    grid_weights = grid_weights[:seq_len, :seq_len]
    
    # Plotting code
    fig, ax = plt.subplots(figsize=(max(5, seq_len * 0.4), max(5, seq_len * 0.4)))
    im = ax.imshow(grid_weights, cmap='plasma')
    
    ax.set_xticks(np.arange(seq_len))
    ax.set_yticks(np.arange(seq_len))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    
    # Rotate token text labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_title(f"Attention Heatmap: Layer {layer_idx + 1}, Head {head_idx + 1}")
    fig.tight_layout()
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.savefig(save_path, dpi=150)
    plt.close()


def load_large_dataset(dataset_name_or_path: str, max_chars: int = None, streaming: bool = False, vocab: list = None, dataset_config: str = None) -> str:
    """
    Loads a text dataset from Hugging Face Datasets Hub or a local text file.
    Concatenates text entries and returns a single large text string, capped at max_chars.
    """
    import os
    
    # 1. Handle local file reading directly (fast path)
    if dataset_name_or_path and os.path.exists(dataset_name_or_path):
        print(f"Loading local file '{dataset_name_or_path}'...")
        try:
            with open(dataset_name_or_path, 'r', encoding='utf-8', errors='ignore') as f:
                if max_chars is not None:
                    text = f.read(max_chars)
                else:
                    text = f.read()
            print(f"Loaded {len(text)} characters from local file.")
            return text
        except Exception as e:
            print(f"Error reading local file: {e}. Trying via HF datasets...")

    # 2. Try loading as Hugging Face dataset
    try:
        from datasets import load_dataset
    except ImportError:
        print("WARNING: Hugging Face 'datasets' library is not installed. Falling back to local file 'shakespeare.txt'...")
        # Check if local file exists
        if os.path.exists("shakespeare.txt"):
            with open("shakespeare.txt", "r", encoding="utf-8", errors="ignore") as f:
                text = f.read(max_chars) if max_chars else f.read()
            return text
        else:
            raise ImportError("Hugging Face 'datasets' library is required to load remote datasets. Run 'pip install datasets'.")

    print(f"Loading HF dataset '{dataset_name_or_path}' (streaming={streaming})...")
    
    try:
        # Determine dataset name and potential subset
        if dataset_config:
            ds = load_dataset(dataset_name_or_path, dataset_config, split="train", streaming=streaming)
        elif dataset_name_or_path == "wikitext" or dataset_name_or_path == "wikitext-2":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=streaming)
        elif dataset_name_or_path == "wikitext-103":
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=streaming)
        elif dataset_name_or_path == "Salesforce/wikitext":
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=streaming)
        else:
            ds = load_dataset(dataset_name_or_path, split="train", streaming=streaming)
    except Exception as e:
        print(f"Failed to load dataset '{dataset_name_or_path}' via datasets: {e}. Trying fallback 'shakespeare.txt'...")
        if os.path.exists("shakespeare.txt"):
            with open("shakespeare.txt", "r", encoding="utf-8", errors="ignore") as f:
                text = f.read(max_chars) if max_chars else f.read()
            return text
        raise e

    # Concatenate the text rows
    texts = []
    current_chars = 0
    
    for row in ds:
        # Detect the text field dynamically
        text_val = None
        for key in ["text", "content", "story", "document"]:
            if key in row:
                text_val = row[key]
                break
        
        if text_val is None:
            # Fallback to the first string value in the row dictionary
            for val in row.values():
                if isinstance(val, str):
                    text_val = val
                    break
                    
        if text_val:
            texts.append(text_val)
            current_chars += len(text_val)
            if max_chars is not None and current_chars >= max_chars:
                break
                
    full_text = "".join(texts)
    if max_chars is not None:
        full_text = full_text[:max_chars]
        
    print(f"Loaded {len(full_text)} characters from '{dataset_name_or_path}'.")
    return full_text
