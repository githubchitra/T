# Demystifying Transformers: Building Attention Mechanisms from Scratch in PyTorch

In recent years, the Transformer architecture has become the undisputed backbone of modern Artificial Intelligence, powering state-of-the-art models like GPT-4, Claude, and Gemini. While high-level libraries like Hugging Face or PyTorch's native `nn.Transformer` make it easy to deploy these models, they abstract away the underlying mechanics. 

To truly understand how Transformers process sequences, we built a complete Transformer architecture completely from scratch using PyTorch (no high-level pre-built attention layers allowed). In this post, we'll walk through the implementation choices, the mathematics, and the lessons learned.

---

## 1. The Core Engine: Scaled Dot-Product Attention

At the heart of the Transformer is the **Scaled Dot-Product Attention**. It allows the model to dynamic focus on different parts of an input sequence, regardless of their distance.

Given three representations—**Queries ($Q$)**, **Keys ($K$)**, and **Values ($V$)**—attention computes similarity scores between the Queries and Keys. These scores determine how much weight is given to each Value.

The mathematical definition is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

### Implementation Details:
```python
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
if mask is not None:
    scores = scores.masked_fill(mask == 0, float('-inf'))
attn_weights = torch.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, v)
```

### Why Scale by $\sqrt{d_k}$?
Without the scaling factor $\sqrt{d_k}$, as the dimension of the keys $d_k$ grows, the dot products grow large in magnitude. This pushes the softmax function into regions with extremely small gradients (the vanishing gradient problem). Dividing by $\sqrt{d_k}$ stabilizes training.

---

## 2. Multi-Head Attention: Broadening Horizons

Instead of performing attention once over the full embedding dimension, we split the queries, keys, and values into $h$ different "heads" and project them to a lower-dimensional space ($d_k = d_{model} / h$). 

Each head performs attention independently in parallel. This allows the model to jointly attend to information from different representation subspaces at different positions. Finally, the outputs are concatenated and projected back to the original dimension.

```python
# Split heads: view and transpose
q_proj = self.q_proj(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
# Apply scaled dot-product attention ...
# Merge heads back: transpose and view
context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
output = self.out_proj(context)
```

---

## 3. Pre-LN vs. Post-LN: A Key Architectural Choice

In the original "Attention Is All You Need" paper, Layer Normalization was applied *after* the residual additions (Post-LN):

$$x_{l+1} = \text{LayerNorm}(x_l + \text{SubLayer}(x_l))$$

However, training deep Post-LN architectures is notoriously unstable and requires a highly specific learning rate warmup phase to prevent early divergence. 

For this project, we implemented **Pre-Layer Normalization (Pre-LN)**, which normalizes inputs before they enter the sublayers:

$$x_{l+1} = x_l + \text{SubLayer}(\text{LayerNorm}(x_l))$$

Pre-LN allows gradients to flow directly through the residual stream from the final layer back to the first layer, leading to more stable initialization and training, and eliminating the extreme sensitivity to learning rate warmups.

---

## 4. Sequence Modeling: Transformer vs. Custom LSTM

To appreciate what Transformers bring to the table, we also implemented an **LSTM model from scratch** to serve as a baseline. The LSTM updates its hidden state $h_t$ and cell state $c_t$ sequentially step-by-step using gating equations:

$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c x_t + U_c h_{t-1} + b_c)$$
$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$
$$h_t = o_t \odot \tanh(c_t)$$

### Key Differences:
- **Parallelism**: LSTMs must process tokens sequentially ($O(N)$ sequential operations), while Transformers process all tokens at once ($O(1)$ sequential operations).
- **Context Length**: LSTMs struggle with long-term dependencies because information must pass through many sequential updates. Transformers can access any position in constant time, though they pay a quadratic cost $O(N^2)$ in self-attention compute.

---

## 5. Lessons Learned & Verification

1. **Precision Matters in Masking**: Masking is easily prone to bugs. During early testing, it is crucial to ensure that attention weights for future tokens are strictly zero, which we verified with our synthetic repeating pattern test (`"abcdabcd..."`), where the model achieved $>90\%$ accuracy on held-out text indices.
2. **Weight Initialization**: Standard PyTorch initializations are often suboptimal for Transformers. Applying **Xavier Uniform** initialization to linear layers and a small standard deviation **Normal (0.02)** to embeddings was crucial for stable training dynamics.
3. **Warmup & Decay**: The **Noam learning rate scheduler** is essential when training with higher learning rates. Starting with a linear warmup, then decaying proportionally to $\text{step}^{-0.5}$, helps the model search the parameter space early on without exploding.
