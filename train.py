import argparse
import time
import os
import urllib.request
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import custom modules
from attention import ScaledDotProductAttention, MultiHeadAttention
from transformer_block import PositionalEncoding, EncoderBlock, DecoderBlock
from model import TransformerLM, BaselineLSTM, TransformerSeq2Seq
from utils import CharDataset, NoamScheduler, generate_text, plot_attention_heatmap, load_large_dataset


def get_shakespeare_data() -> str:
    """
    Downloads the Tiny Shakespeare dataset if it is not already present locally.
    Falls back to a synthetic story if download fails.
    """
    file_path = "shakespeare.txt"
    if not os.path.exists(file_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Dataset 'shakespeare.txt' not found. Downloading from {url}...")
        try:
            urllib.request.urlretrieve(url, file_path)
            print("Download completed successfully.")
        except Exception as e:
            print(f"Failed to download from GitHub: {e}. Writing fallback data...")
            fallback_text = (
                "Once upon a time, in a land far far away, there lived a wise king who wanted "
                "to understand the mysteries of attention mechanisms. He gathered the greatest "
                "scholars in the world to build a machine from scratch. The scholars worked day "
                "and night, crafting matrices of queries, keys, and values, until at last they "
                "had created a model that could think and write like a human. "
            ) * 1000
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(fallback_text)
            print("Fallback data written to shakespeare.txt.")
    return file_path


def run_synthetic_test(device: torch.device) -> bool:
    """
    Executes the synthetic test: trains on the repeating pattern 'abcdabcd...'
    from positions 0-799, and evaluates next-token predictions on positions 800-999.
    Target accuracy: > 90%.
    """
    print("\n" + "=" * 50)
    print("RUNNING SYNTHETIC TEST (Target Accuracy: >90%)")
    print("=" * 50)
    
    # 1. Define sequence
    pattern = "abcd"
    full_seq = pattern * 250  # 1000 characters
    
    train_text = full_seq[:800]
    test_text = full_seq[800:1000]
    
    # 2. Tokenizer setup
    vocab = sorted(list(set(pattern)))
    char_to_ix = {ch: i for i, ch in enumerate(vocab)}
    ix_to_char = {i: ch for i, ch in enumerate(vocab)}
    
    # 3. Create datasets (sliding windows)
    seq_len = 8
    
    def get_windows(indices, seq_len):
        x_list, y_list = [], []
        for i in range(len(indices) - seq_len):
            x_list.append(indices[i : i + seq_len])
            y_list.append(indices[i + 1 : i + seq_len + 1])
        return torch.tensor(x_list, dtype=torch.long), torch.tensor(y_list, dtype=torch.long)
        
    train_indices = [char_to_ix[ch] for ch in train_text]
    test_indices = [char_to_ix[ch] for ch in test_text]
    
    X_train, Y_train = get_windows(train_indices, seq_len)
    X_test, Y_test = get_windows(test_indices, seq_len)
    
    # 4. Model instantiation
    # Use a small configuration for rapid training
    model = TransformerLM(
        vocab_size=len(vocab),
        embed_dim=32,
        num_layers=2,
        num_heads=2,
        d_ff=64,
        max_len=50,
        dropout=0.0
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training loop
    model.train()
    batch_size = 32
    num_samples = X_train.size(0)
    
    print("Training tiny Transformer model on 'abcd' repeating pattern...")
    for epoch in range(25):
        permutation = torch.randperm(num_samples)
        epoch_loss = 0
        num_batches = 0
        for i in range(0, num_samples, batch_size):
            indices = permutation[i : i + batch_size]
            bx = X_train[indices].to(device)
            by = Y_train[indices].to(device)
            
            optimizer.zero_grad()
            logits, _ = model(bx)  # [Batch, SeqLen, Vocab]
            loss = criterion(logits.view(-1, len(vocab)), by.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/25 - Loss: {epoch_loss / num_batches:.4f}")
            
    # 6. Evaluation
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)
        logits, _ = model(X_test)
        predictions = torch.argmax(logits, dim=-1)
        
        correct = (predictions == Y_test).sum().item()
        total = Y_test.numel()
        accuracy = correct / total
        
        print("-" * 50)
        print(f"Synthetic Test Sequence Accuracy (Positions 800-999): {accuracy * 100:.2f}%")
        
        if accuracy >= 0.90:
            print("RESULT: SUCCESS (Accuracy >= 90%)")
            print("=" * 50 + "\n")
            return True
        else:
            print("RESULT: FAILURE (Accuracy < 90%)")
            print("=" * 50 + "\n")
            return False


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                optimizer: torch.optim.Optimizer, scheduler: NoamScheduler, 
                epochs: int, device: torch.device, val_interval: int = 1,
                accumulate_steps: int = 1, save_path: str = None,
                vocab: list = None, config: dict = None,
                mixed_precision: bool = False,
                # New parameters for extended capabilities
                use_amp: bool = None,
                gradient_accumulation_steps: int = None,
                save_dir: str = None,
                save_every: int = 10,
                start_epoch: int = 0,
                best_val_loss: float = float('inf')) -> tuple:
    """
    Standard training loop for a language model (TransformerLM or BaselineLSTM) with gradient accumulation and checkpoint saving.
    Supports mixed precision (AMP) when mixed_precision=True and CUDA is available.
    """
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    
    # Map arguments for backward compatibility
    final_use_amp = use_amp if use_amp is not None else mixed_precision
    final_accumulate_steps = gradient_accumulation_steps if gradient_accumulation_steps is not None else accumulate_steps
    final_save_dir = save_dir if save_dir is not None else (os.path.dirname(save_path) if save_path else "checkpoints")
    
    if final_save_dir:
        os.makedirs(final_save_dir, exist_ok=True)
        
    # Configure AMP
    use_amp_cuda = final_use_amp and (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp_cuda)
    
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp_cuda):
                logits, _ = model(x)
                # Flatten to compute loss
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                # Scale loss for gradient accumulation
                scaled_loss = loss / final_accumulate_steps
            
            scaler.scale(scaled_loss).backward()
            
            # Update weights after accumulating gradients
            if (batch_idx + 1) % final_accumulate_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})
            
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if (epoch + 1) % val_interval == 0:
            model.eval()
            total_val_loss = 0
            num_val_batches = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.cuda.amp.autocast(enabled=use_amp_cuda):
                        v_logits, _ = model(vx)
                        v_loss = criterion(v_logits.view(-1, v_logits.size(-1)), vy.view(-1))
                    total_val_loss += v_loss.item()
                    num_val_batches += 1
                    
            avg_val_loss = total_val_loss / num_val_batches
            val_losses.append(avg_val_loss)
            
            # Prepare checkpoint dict
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': vocab,
                'config': config,
                'best_val_loss': best_val_loss,
                'scheduler_step_num': scheduler.step_num if scheduler is not None else 0
            }
            
            # Checkpoint saving when validation loss improves
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint['best_val_loss'] = best_val_loss
                
                # Save best_model.pt in save_dir
                best_path = os.path.join(final_save_dir, "best_model.pt")
                torch.save(checkpoint, best_path)
                print(f"Epoch {epoch+1} Completed - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} (Saved new best model to {best_path})")
                
                # Also support legacy save_path if passed
                if save_path:
                    torch.save(checkpoint, save_path)
            else:
                print(f"Epoch {epoch+1} Completed - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                
            # Save periodic checkpoints
            if (epoch + 1) % save_every == 0:
                periodic_path = os.path.join(final_save_dir, f"checkpoint_epoch_{epoch+1}.pt")
                torch.save(checkpoint, periodic_path)
                print(f"Saved periodic checkpoint: {periodic_path}")
        else:
            val_losses.append(None)
            print(f"Epoch {epoch+1} Completed - Train Loss: {avg_train_loss:.4f}")
            
    return train_losses, val_losses


def get_latest_checkpoint(save_dir: str):
    """
    Scans the save_dir for files matching 'checkpoint_epoch_*.pt' and returns the path
    to the one with the highest epoch number. Falls back to best_model.pt if none found.
    """
    if not os.path.exists(save_dir):
        return None
    files = [f for f in os.listdir(save_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
    if not files:
        best_model_path = os.path.join(save_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            return best_model_path
        return None
    
    checkpoint_epochs = []
    for f in files:
        try:
            epoch_num = int(f.split("_")[-1].split(".")[0])
            checkpoint_epochs.append((epoch_num, f))
        except ValueError:
            continue
            
    if not checkpoint_epochs:
        best_model_path = os.path.join(save_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            return best_model_path
        return None
        
    latest_file = max(checkpoint_epochs, key=lambda x: x[0])[1]
    return os.path.join(save_dir, latest_file)


def main():
    parser = argparse.ArgumentParser(description="Train custom Transformer / LSTM models from scratch.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Starting learning rate (if Noam scheduler is off)")
    parser.add_argument("--noam", action="store_true", default=True, help="Use Noam learning rate scheduler")
    parser.add_argument("--no_noam", dest="noam", action="store_false", help="Disable Noam learning rate scheduler")
    parser.add_argument("--warmup", type=int, default=2000, help="Warmup steps for Noam scheduler")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (defaults to 128 for GPU, 64 for CPU)")
    parser.add_argument("--seq_len", type=int, default=128, help="Context sequence length")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of transformer/LSTM layers")
    parser.add_argument("--d_ff", type=int, default=512, help="Feedforward network dimension size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--test-synthetic", action="store_true", help="Run the synthetic testing pattern verification")
    parser.add_argument("--tiny", action="store_true", help="Use a tiny configuration for faster execution on CPU")
    parser.add_argument("--baseline", action="store_true", help="Train a baseline LSTM model to compare performance")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt string for character generation")
    parser.add_argument("--gen_len", type=int, default=400, help="Number of characters to generate")
    parser.add_argument("--temperature", "--temp", type=float, default=0.8, help="Temperature for text generation sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling threshold")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling threshold")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty factor")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--load_best", action="store_true", help="Load the best model checkpoint for generation")
    parser.add_argument("--accumulate_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # New CLI arguments
    parser.add_argument("--dataset_name", type=str, default=None, help="HF dataset name or local path (if None, falls back to shakespeare.txt)")
    parser.add_argument("--dataset_config", type=str, default=None, help="HF dataset config name (e.g., 'wikitext-2-raw-v1')")
    parser.add_argument("--max_chars", type=int, default=10000000, help="Limit dataset size")
    parser.add_argument("--mixed_precision", action="store_true", default=None, help="Use mixed precision training")
    parser.add_argument("--no_mixed_precision", dest="mixed_precision", action="store_false", help="Disable mixed precision training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in save_dir")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Run synthetic pattern correctness verification if specified
    if args.test_synthetic:
        success = run_synthetic_test(device)
        return
        
    # Dynamically resolve default batch size and mixed precision
    if args.batch_size is None:
        args.batch_size = 128 if torch.cuda.is_available() else 64
        
    if args.mixed_precision is None:
        args.mixed_precision = torch.cuda.is_available()
        
    # Support both accumulate_steps and gradient_accumulation_steps for backward compatibility
    accumulate_steps = max(args.accumulate_steps, args.gradient_accumulation_steps)
    
    checkpoint = None
    start_epoch = 0
    best_val_loss = float('inf')
    vocab = None
    config = None
    
    # Handle resuming
    if args.resume:
        latest_checkpoint_path = get_latest_checkpoint(args.save_dir)
        if latest_checkpoint_path:
            print(f"Resuming training: Loading checkpoint from '{latest_checkpoint_path}'...")
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            start_epoch = checkpoint.get('epoch', 0)
            vocab = checkpoint.get('vocab', None)
            config = checkpoint.get('config', None)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed from epoch {start_epoch} | Best Val Loss: {best_val_loss:.4f}")
        else:
            print("WARNING: Resume active but no checkpoints found in save_dir. Starting from scratch.")
            
    # 3. Load dataset
    if args.dataset_name is None:
        dataset_path = get_shakespeare_data()
        with open(dataset_path, "r", encoding="utf-8") as f:
            text = f.read(args.max_chars) if args.max_chars else f.read()
    else:
        text = load_large_dataset(args.dataset_name, max_chars=args.max_chars, dataset_config=args.dataset_config)
        
    # 4. Fallback for CPU / Speed (tiny mode overrides)
    if args.tiny:
        text = text[:50000]  # First 50KB only
        args.embed_dim = 64
        args.num_layers = 2
        args.num_heads = 4
        args.d_ff = 256
        args.epochs = min(args.epochs, 3)
        args.batch_size = 64
        accumulate_steps = 4
        print("--- TINY CONFIG FALLBACK APPLIED ---")
        print(f"Dataset length: {len(text)} characters")
        print(f"Embedding dimension: {args.embed_dim}, Layers: {args.num_layers}, Heads: {args.num_heads}")
        print(f"Batch size: {args.batch_size}, Gradient accumulation steps: {accumulate_steps}")
        print("-----------------------------------")
        
    print(f"Dataset Loaded. Total characters: {len(text)}")
    dataset = CharDataset(text, args.seq_len, chars=vocab)
    vocab_size = dataset.vocab_size
    print(f"Vocabulary Size: {vocab_size} unique characters")
    
    # 5. Train / Val split (90% training, 10% validation)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    # Reconstruct hyperparams if config loaded
    if config is not None:
        print("Restoring model configuration from checkpoint...")
        model_embed_dim = config.get('embed_dim', args.embed_dim)
        model_num_layers = config.get('num_layers', args.num_layers)
        model_num_heads = config.get('num_heads', args.num_heads)
        model_d_ff = config.get('d_ff', args.d_ff)
        model_seq_len = config.get('seq_len', args.seq_len)
        model_dropout = config.get('dropout', args.dropout)
    else:
        model_embed_dim = args.embed_dim
        model_num_layers = args.num_layers
        model_num_heads = args.num_heads
        model_d_ff = args.d_ff
        model_seq_len = args.seq_len
        model_dropout = args.dropout
        
    # 6. Initialize Transformer model
    print(f"\nInitializing custom Decoder-Only TransformerLM model...")
    transformer_model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=model_embed_dim,
        num_layers=model_num_layers,
        num_heads=model_num_heads,
        d_ff=model_d_ff,
        max_len=model_seq_len + 50,
        dropout=model_dropout
    ).to(device)
    transformer_model.seq_len = model_seq_len
    
    # Load model weights if resuming
    if checkpoint is not None:
        transformer_model.load_state_dict(checkpoint['model_state_dict'])
        print("Restored model weights from checkpoint.")
        
    # 7. Optimizer & Scheduler setup
    t_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=args.lr if not args.noam else 1.0)
    if checkpoint is not None and 'optimizer_state_dict' in checkpoint:
        t_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Restored optimizer state from checkpoint.")
        
    t_scheduler = None
    if args.noam:
        t_scheduler = NoamScheduler(t_optimizer, embed_dim=model_embed_dim, warmup_steps=args.warmup)
        if checkpoint is not None and 'scheduler_step_num' in checkpoint:
            t_scheduler.step_num = checkpoint['scheduler_step_num']
            print(f"Restored scheduler step number to {t_scheduler.step_num}.")
            
    # 8. Train Transformer
    t_start = time.time()
    t_config = {
        'vocab_size': vocab_size,
        'embed_dim': model_embed_dim,
        'num_layers': model_num_layers,
        'num_heads': model_num_heads,
        'd_ff': model_d_ff,
        'seq_len': model_seq_len,
        'dropout': model_dropout
    }
    t_save_path = os.path.join(args.save_dir, "best_transformer.pt") if args.save_dir else None
    
    t_train_losses, t_val_losses = train_model(
        model=transformer_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=t_optimizer,
        scheduler=t_scheduler,
        epochs=args.epochs,
        device=device,
        accumulate_steps=accumulate_steps,
        save_path=t_save_path,
        vocab=dataset.chars,
        config=t_config,
        mixed_precision=args.mixed_precision,
        # New parameters
        use_amp=args.mixed_precision,
        gradient_accumulation_steps=accumulate_steps,
        save_dir=args.save_dir,
        save_every=args.save_every,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss
    )
    t_duration = time.time() - t_start
    print(f"Transformer training completed in {t_duration:.2f} seconds.")
    
    # Load best model for generation if requested
    if args.load_best or args.resume:
        best_path = os.path.join(args.save_dir, "best_model.pt")
        if not os.path.exists(best_path):
            best_path = os.path.join(args.save_dir, "best_transformer.pt")
            
        if os.path.exists(best_path):
            print(f"Loading best Transformer model checkpoint from '{best_path}' for generation.")
            best_checkpoint = torch.load(best_path, map_location=device)
            if isinstance(best_checkpoint, dict) and 'model_state_dict' in best_checkpoint:
                transformer_model.load_state_dict(best_checkpoint['model_state_dict'])
                if 'config' in best_checkpoint and 'seq_len' in best_checkpoint['config']:
                    transformer_model.seq_len = best_checkpoint['config']['seq_len']
            else:
                transformer_model.load_state_dict(best_checkpoint)
                
    # 9. Evaluate & Autoregressively generate sample text from Transformer
    print(f"\n{'='*20} TRANSFORMER GENERATED TEXT {'='*20}")
    generated_transformer = generate_text(
        model=transformer_model,
        prompt=args.prompt,
        vocab=dataset.chars,
        max_new_tokens=args.gen_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device
    )
    print(generated_transformer)
    print("=" * 68 + "\n")
    
    # 10. Save attention maps from Transformer
    print("Generating attention map heatmap for the prompt...")
    transformer_model.eval()
    
    plot_prompt = args.prompt
    if plot_prompt and not plot_prompt.endswith(" "):
        plot_prompt = plot_prompt + " "
        
    prompt_tokens = [dataset.char_to_ix.get(ch, 0) for ch in plot_prompt]
    with torch.no_grad():
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        _, attn_weights = transformer_model(prompt_tensor)
        
        os.makedirs("plots", exist_ok=True)
        heatmap_path = "plots/attention_map.png"
        plot_attention_heatmap(attn_weights, plot_prompt, layer_idx=0, head_idx=0, save_path=heatmap_path)
        print(f"Attention map saved to '{heatmap_path}'.")
        
    # 11. Train Baseline LSTM model if requested
    l_train_losses, l_val_losses = [], []
    generated_lstm = ""
    if args.baseline:
        print(f"\nInitializing custom Baseline LSTM model...")
        lstm_model = BaselineLSTM(
            vocab_size=vocab_size,
            embed_dim=model_embed_dim,
            hidden_dim=model_embed_dim,
            num_layers=model_num_layers
        ).to(device)
        lstm_model.seq_len = model_seq_len
        
        # LSTM uses standard Adam with fixed learning rate
        l_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=args.lr)
        
        l_config = {
            'vocab_size': vocab_size,
            'embed_dim': model_embed_dim,
            'hidden_dim': model_embed_dim,
            'num_layers': model_num_layers,
            'seq_len': model_seq_len
        }
        l_save_path = os.path.join(args.save_dir, "best_lstm.pt") if args.save_dir else None
        
        l_start = time.time()
        print("Training Baseline LSTM model...")
        l_train_losses, l_val_losses = train_model(
            model=lstm_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=l_optimizer,
            scheduler=None,
            epochs=args.epochs,
            device=device,
            accumulate_steps=accumulate_steps,
            save_path=l_save_path,
            vocab=dataset.chars,
            config=l_config,
            mixed_precision=args.mixed_precision,
            # New parameters
            use_amp=args.mixed_precision,
            gradient_accumulation_steps=accumulate_steps,
            save_dir=args.save_dir,
            save_every=args.save_every,
            start_epoch=start_epoch,
            best_val_loss=best_val_loss
        )
        l_duration = time.time() - l_start
        print(f"LSTM training completed in {l_duration:.2f} seconds.")
        
        # Load best LSTM model if requested
        if args.load_best or args.resume:
            best_lstm_path = os.path.join(args.save_dir, "best_model_lstm.pt") # separate best LSTM check
            if not os.path.exists(best_lstm_path) and l_save_path:
                best_lstm_path = l_save_path
            if os.path.exists(best_lstm_path):
                print(f"Loading best LSTM model checkpoint from '{best_lstm_path}' for generation.")
                best_lstm_checkpoint = torch.load(best_lstm_path, map_location=device)
                if isinstance(best_lstm_checkpoint, dict) and 'model_state_dict' in best_lstm_checkpoint:
                    lstm_model.load_state_dict(best_lstm_checkpoint['model_state_dict'])
                else:
                    lstm_model.load_state_dict(best_lstm_checkpoint)
                    
        print(f"\n{'='*20} LSTM GENERATED TEXT {'='*20}")
        generated_lstm = generate_text(
            model=lstm_model,
            prompt=args.prompt,
            vocab=dataset.chars,
            max_new_tokens=args.gen_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device
        )
        print(generated_lstm)
        print("=" * 60 + "\n")
        
    # 12. Plot loss curves
    plt.figure(figsize=(10, 5))
    epochs_range = range(start_epoch + 1, args.epochs + 1)
    
    if len(epochs_range) > 0 and len(t_train_losses) == len(epochs_range):
        plt.plot(epochs_range, t_train_losses, label="Transformer Train Loss", color="blue", linestyle="-")
        valid_t_val_losses = [v for v in t_val_losses if v is not None]
        if len(valid_t_val_losses) == len(t_train_losses):
            plt.plot(epochs_range, t_val_losses, label="Transformer Val Loss", color="cyan", linestyle="--")
            
        if args.baseline and len(l_train_losses) == len(epochs_range):
            plt.plot(epochs_range, l_train_losses, label="LSTM Train Loss", color="red", linestyle="-")
            valid_l_val_losses = [v for v in l_val_losses if v is not None]
            if len(valid_l_val_losses) == len(l_train_losses):
                plt.plot(epochs_range, l_val_losses, label="LSTM Val Loss", color="orange", linestyle="--")
                
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Comparison")
        plt.legend()
        plt.grid(True)
        
        loss_plot_path = "plots/loss_curve.png"
        plt.savefig(loss_plot_path, dpi=150)
        plt.close()
        print(f"Loss curves saved to '{loss_plot_path}'.")
        
    # Save a small text report summary
    report_path = "experiment_summary.md"
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("# Experiment Run Summary\n\n")
        rf.write(f"- **Runtime Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        rf.write(f"- **Device**: {device}\n")
        rf.write(f"- **Config**: Embed Dim: {model_embed_dim}, Layers: {model_num_layers}, Heads: {model_num_heads}\n\n")
        rf.write("## Results\n\n")
        rf.write("| Model | Train Duration | Final Train Loss | Final Val Loss |\n")
        rf.write("| --- | --- | --- | --- |\n")
        if len(t_train_losses) > 0:
            rf.write(f"| Transformer | {t_duration:.2f}s | {t_train_losses[-1]:.4f} | {t_val_losses[-1] if t_val_losses[-1] is not None else 'N/A'} |\n")
        if args.baseline and len(l_train_losses) > 0:
            rf.write(f"| LSTM | {l_duration:.2f}s | {l_train_losses[-1]:.4f} | {l_val_losses[-1] if l_val_losses[-1] is not None else 'N/A'} |\n")
        rf.write("\n## Sample Generation Outputs\n\n")
        rf.write("### Transformer Output\n")
        rf.write(f"```\n{generated_transformer}\n```\n\n")
        if args.baseline:
            rf.write("### LSTM Output\n")
            rf.write(f"```\n{generated_lstm}\n```\n")
            
    print(f"Experiment summary written to '{report_path}'.")


if __name__ == "__main__":
    main()
