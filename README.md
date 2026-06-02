# Transformer from Scratch in PyTorch

A clean, modular, and fully documented character-level Transformer language model implemented from scratch using PyTorch (no high-level pre-built layers like `torch.nn.Transformer` or `torch.nn.MultiheadAttention`).

This codebase also includes a custom **LSTM model from scratch** (using standard cell gate equations) to serve as a baseline comparison.

**Live Demo:** [https://transformer-from-scratch.streamlit.app/](https://transformer-from-scratch.streamlit.app/)

## Architecture & Mathematical Formulations

### 1. Scaled Dot-Product Attention
Attention maps queries ($Q$), keys ($K$), and values ($V$) to outputs. The similarity score is scaled by the square root of the key dimension ($d_k$) to prevent gradients from vanishing in softmax:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

where $M$ is an optional attention mask (e.g. causal triangular mask to block lookahead).

### 2. Multi-Head Attention
Instead of performing a single attention function, Multi-Head Attention linearly projects $Q$, $K$, and $V$ to lower dimensions $h$ times, performs Scaled Dot-Product Attention in parallel, concatenates the outputs, and projects them back:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$
$$\text{where} \quad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 3. Positional Encoding
Since Transformers have no recurrence or convolution, we inject order information using sinusoidal waves of varying frequencies:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

---

## File Structure

- `attention.py`: `ScaledDotProductAttention` and `MultiHeadAttention` implementation from basic linear mappings.
- `transformer_block.py`: `PositionalEncoding`, `PositionwiseFeedForward`, Pre-LN `EncoderBlock` and `DecoderBlock`.
- `model.py`: Full `TransformerEncoder`, `TransformerDecoder`, Seq2Seq `TransformerSeq2Seq`, Decoder-Only language model `TransformerLM`, and custom `BaselineLSTM`.
- `utils.py`: Text generation helper (`generate_text`), Noam learning rate scheduler (`NoamScheduler`), `CharDataset`, and attention heatmap plotting (`plot_attention_heatmap`).
- `train.py`: Main trainer orchestrator. Handles training loops, validation, CLI args, synthetic patterns verification, and comparison plotting.

---

## Installation & Setup

1. Make sure Python 3.8+ is installed.
2. Clone the repository and navigate into it.
3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Running the Code

### 1. Run Verification (Synthetic Test)
Trains a tiny model on the repeating pattern `"abcdabcd..."` using positions 0-799, and verifies that the character predictions on positions 800-999 exceed **90% accuracy**.
```bash
python train.py --test-synthetic
```

### 2. Train on Shakespeare Dataset (Tiny CPU Quick Run)
Trains the model on a tiny 50KB subset of Shakespeare with a lightweight configuration (embedding dim 64, 2 layers, 4 heads) to finish in less than a minute on CPU:
```bash
python train.py --tiny --epochs 3
```

### 3. Standard Training & Baseline Comparison
Trains both the custom **Transformer** and **Baseline LSTM** on the full Tiny Shakespeare dataset, plots comparative loss curves, and generates attention maps:
```bash
python train.py --epochs 5 --embed_dim 128 --num_heads 4 --num_layers 3 --baseline
```

---

## Outputs & Visualization

All visualizations are stored in the `./plots/` directory:
- `plots/loss_curve.png`: Comparative loss curves between Transformer and LSTM (if `--baseline` was run).
- `plots/attention_map.png`: Heatmap of attention weights of the first layer and first head on the evaluation prompt.
- `experiment_summary.md`: Detailed markdown table of training times, loss convergence, and generated text samples.
