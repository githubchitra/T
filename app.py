import os
import time
import torch
import streamlit as st
from model import TransformerLM
from utils import generate_text

# Set page config
st.set_page_config(
    page_title="Transformer LM Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS Injection
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@300;400;600;700&display=swap');

/* Apply modern typography */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif;
}

/* Gradient Title */
.main-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    margin-top: -1rem;
}

.subtitle {
    font-family: 'Outfit', sans-serif;
    font-size: 1.15rem;
    text-align: center;
    color: #94a3b8;
    margin-bottom: 2rem;
}

/* Styled Container Cards */
.metric-card {
    background-color: #1e293b;
    border-radius: 12px;
    padding: 1.25rem;
    border: 1px solid #334155;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.metric-title {
    font-size: 0.9rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f8fafc;
}

/* Text output styling */
.generated-box-container {
    background-color: #0f172a;
    border-radius: 10px;
    padding: 1.5rem;
    border-left: 6px solid #a855f7;
    margin-top: 1rem;
    box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
}

.generated-text {
    font-family: 'Courier New', Courier, monospace;
    font-size: 1.15rem;
    line-height: 1.6;
    color: #e2e8f0;
    white-space: pre-wrap;
}

/* Sidebar Custom Styling */
.sidebar-section {
    font-weight: 600;
    color: #a855f7;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<div class='main-title'>🧠 Antigravity Transformer LM Playground</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Interact with a character-level Decoder-Only Transformer built entirely from scratch in PyTorch.</div>", unsafe_allow_html=True)

# Helper function to load model safely
@st.cache_resource
def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None, f"Checkpoint file not found at '{checkpoint_path}'."
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'config' in checkpoint and 'model_state_dict' in checkpoint:
            config = checkpoint['config']
            vocab = checkpoint['vocab']
            state_dict = checkpoint['model_state_dict']
            
            # Reconstruct the Transformer model dynamically using saved config
            model = TransformerLM(
                vocab_size=config['vocab_size'],
                embed_dim=config['embed_dim'],
                num_layers=config['num_layers'],
                num_heads=config['num_heads'],
                d_ff=config['d_ff'],
                max_len=config.get('seq_len', 128) + 50,
                dropout=config.get('dropout', 0.0)
            )
            model.load_state_dict(state_dict)
            model.seq_len = config.get('seq_len', 128)
            model.eval()
            return (model, vocab, config), None
        else:
            return None, "Invalid checkpoint format. Ensure it was saved as a dict with 'model_state_dict', 'vocab', and 'config'."
    except Exception as e:
        return None, f"Error reading checkpoint: {str(e)}"

# Sidebar for Model Configuration & File Selection
st.sidebar.markdown("### 🛠️ Model Source")
default_checkpoint_path = "checkpoints/best_model.pt"
checkpoint_path = st.sidebar.text_input("Checkpoint Path", value=default_checkpoint_path)

# Try loading the model
model_data, load_error = load_model(checkpoint_path)

if load_error:
    st.error(load_error)
    st.info("💡 To train a model and save a checkpoint, run:\n`python train.py --tiny --save_dir checkpoints`")
    st.stop()

model, vocab, config = model_data

# Display Model Architecture Specs in Columns
st.markdown("### 📊 Model Architecture Specs")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Embedding Dim</div><div class='metric-value'>{config['embed_dim']}</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Encoder Layers</div><div class='metric-value'>{config['num_layers']}</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Attention Heads</div><div class='metric-value'>{config['num_heads']}</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Vocab Size</div><div class='metric-value'>{config['vocab_size']}</div></div>", unsafe_allow_html=True)

st.write("")

# Layout for Generation settings & Input
st.sidebar.markdown("### 🎛️ Generation Controls")
temperature = st.sidebar.slider("Temperature (Creativity)", min_value=0.1, max_value=2.0, value=0.8, step=0.05)
top_k = st.sidebar.slider("Top-k Filtering", min_value=1, max_value=100, value=50, step=1)
top_p = st.sidebar.slider("Top-p (Nucleus) Filtering", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
repetition_penalty = st.sidebar.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.2, step=0.05)
max_new_tokens = st.sidebar.slider("Tokens to Generate", min_value=10, max_value=1000, value=200, step=10)

# Main UI layout: Prompt input on the left, static attention map on the right
left_col, right_col = st.columns([3, 2])

with left_col:
    st.markdown("### 📝 Prompt Input")
    prompt = st.text_area("Enter a starter prompt:", value="Once upon a time", height=100)
    
    generate_btn = st.button("✨ Generate Text", use_container_width=True)
    
    if generate_btn:
        with st.spinner("Generating text autoregressively..."):
            start_time = time.time()
            generated_text = generate_text(
                model=model,
                prompt=prompt,
                vocab=vocab,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                device='cpu'
            )
            duration = time.time() - start_time
            
        st.success(f"Generated successfully in {duration:.2f}s!")
        st.markdown("### 📖 Output")
        st.markdown(
            f"<div class='generated-box-container'><div class='generated-text'>{generated_text}</div></div>",
            unsafe_allow_html=True
        )

with right_col:
    st.markdown("### 🔍 Model Training Attention Map")
    static_heatmap_path = "plots/attention_map.png"
    if os.path.exists(static_heatmap_path):
        st.image(static_heatmap_path, caption="Attention map plotted for the training prompt (Layer 1, Head 1).", use_column_width=True)
    else:
        st.info("ℹ️ Attention map plot 'plots/attention_map.png' not found. It will display here once you run train.py to generate it.")
