"""
SongBloom Streamlit Web Interface
Modern web interface for interactive music generation with Streamlit
"""
import os
import sys
import torch
import torchaudio
import streamlit as st
from pathlib import Path
from datetime import datetime
import tempfile

# Add SongBloom directory to path
songbloom_dir = Path(__file__).parent / "SongBloom-master"
sys.path.insert(0, str(songbloom_dir))

# Import SongBloom components
try:
    from SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler
except ImportError:
    st.error("⚠️ SongBloom components not found. Please ensure the SongBloom-master directory is present.")
    st.stop()

# Configure page
st.set_page_config(
    page_title="🎵 SongBloom - AI Music Generation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎵 SongBloom Next-Gen X2</h1>
    <p>AI-Powered Song Generation with Advanced Optimizations</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name="songbloom_full_150s", dtype="float32", quantization=None):
    """Load the SongBloom model with caching"""
    try:
        # Change to SongBloom directory
        os.chdir(songbloom_dir)
        
        # Import optimized inference utilities
        from infer_optimized import load_config, hf_download, optimize_model, apply_quantization
        
        # Download model
        local_dir = "./cache"
        hf_download(model_name, local_dir)
        
        # Load config
        cfg = load_config(f"{local_dir}/{model_name}.yaml", parent_dir=local_dir)
        cfg.max_dur = cfg.max_dur + 20
        
        # Set dtype
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16
        }
        dtype_torch = dtype_map.get(dtype, torch.float32)
        
        # Build model
        model = SongBloom_Sampler.build_from_trainer(cfg, strict=True, dtype=dtype_torch)
        
        # Apply optimizations
        model.diffusion = optimize_model(model.diffusion, 'standard')
        
        # Apply quantization if requested
        if quantization:
            model.diffusion = apply_quantization(model.diffusion, quantization)
        
        # Set generation parameters
        gen_params = cfg.inference if hasattr(cfg, 'inference') else {
            'cfg_coef': 1.5,
            'steps': 50,
            'dit_cfg_type': 'h',
            'use_sampling': True,
            'top_k': 200,
            'max_frames': cfg.max_dur * 25
        }
        model.set_generation_params(**gen_params)
        
        return model, cfg
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def generate_music(model, config, lyrics, prompt_audio_path, cfg_coef=1.5, steps=50, top_k=200):
    """Generate music from lyrics and prompt audio"""
    try:
        # Load prompt audio
        prompt_wav, sr = torchaudio.load(prompt_audio_path)
        
        # Resample if needed
        if sr != model.sample_rate:
            prompt_wav = torchaudio.functional.resample(prompt_wav, sr, model.sample_rate)
        
        # Convert to mono and truncate
        prompt_wav = prompt_wav.mean(dim=0, keepdim=True)
        prompt_wav = prompt_wav[..., :10*model.sample_rate]
        
        # Update generation parameters
        model.set_generation_params(
            cfg_coef=cfg_coef,
            steps=steps,
            top_k=top_k,
            dit_cfg_type='h',
            use_sampling=True,
            max_frames=config.max_dur * 25
        )
        
        # Generate
        with torch.cuda.amp.autocast(enabled=True):
            wav = model.generate(lyrics, prompt_wav)
        
        # Save output to temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("./streamlit_outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"generated_{timestamp}.flac"
        
        torchaudio.save(str(output_path), wav[0].cpu().float(), model.sample_rate)
        
        return str(output_path), None
    except Exception as e:
        return None, f"Error during generation: {str(e)}"


# Sidebar - Model Settings
with st.sidebar:
    st.header("⚙️ Model Settings")
    
    model_name = st.selectbox(
        "Model",
        ["songbloom_full_150s"],
        help="Select the SongBloom model to use"
    )
    
    dtype = st.selectbox(
        "Precision",
        ["float32", "float16", "bfloat16"],
        index=0,
        help="Lower precision = faster but may reduce quality"
    )
    
    quantization = st.selectbox(
        "Quantization",
        [None, "int8", "int4"],
        help="Quantization reduces memory usage"
    )
    
    if st.button("🔄 Load Model"):
        with st.spinner("Loading model... This may take a few minutes on first run."):
            model, config = load_model(model_name, dtype, quantization)
            if model is not None:
                st.session_state['model'] = model
                st.session_state['config'] = config
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model")
    
    st.divider()
    
    st.header("📖 About")
    st.markdown("""
    **SongBloom** is a state-of-the-art AI music generation system that creates 
    full-length songs from lyrics and style prompts.
    
    **Features:**
    - 🎵 Full song generation
    - 🎨 Style transfer from audio
    - ⚡ Optimized inference
    - 🎯 High-quality output
    
    **⚠️ Note:** This app requires GPU resources to run. Streamlit Cloud's free tier 
    may not have sufficient resources. Consider deploying on GPU-enabled infrastructure 
    or running locally with a CUDA-capable GPU.
    """)

# Main content area
st.header("🎼 Generate Music")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    
    lyrics = st.text_area(
        "Lyrics",
        height=300,
        placeholder="Enter your lyrics here...\n\nExample:\nVerse 1:\nIn the morning light\nI see your face...",
        help="Enter the lyrics for your song"
    )
    
    prompt_audio = st.file_uploader(
        "Style Prompt Audio (10 seconds recommended)",
        type=["wav", "mp3", "flac", "ogg"],
        help="Upload an audio file that represents the style/genre you want"
    )
    
    st.subheader("⚙️ Advanced Settings")
    
    cfg_coef = st.slider(
        "Guidance Coefficient (CFG)",
        min_value=0.0,
        max_value=5.0,
        value=1.5,
        step=0.1,
        help="Higher values follow the style prompt more closely"
    )
    
    steps = st.slider(
        "Diffusion Steps",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
        help="More steps = better quality but slower"
    )
    
    top_k = st.slider(
        "Top-K Sampling",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Controls randomness in generation"
    )

with col2:
    st.subheader("Output")
    
    if st.button("🎵 Generate Music", type="primary"):
        # Check if model is loaded
        if 'model' not in st.session_state or 'config' not in st.session_state:
            st.error("❌ Please load the model first using the sidebar!")
        elif not lyrics or lyrics.strip() == "":
            st.error("❌ Please provide lyrics!")
        elif prompt_audio is None:
            st.error("❌ Please provide a prompt audio file!")
        else:
            # Save uploaded audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(prompt_audio.name)[1]) as tmp_file:
                tmp_file.write(prompt_audio.getvalue())
                tmp_path = tmp_file.name
            
            try:
                with st.spinner("🎵 Generating music... This may take a few minutes."):
                    output_path, error = generate_music(
                        st.session_state['model'],
                        st.session_state['config'],
                        lyrics,
                        tmp_path,
                        cfg_coef=cfg_coef,
                        steps=steps,
                        top_k=top_k
                    )
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.success("✅ Music generated successfully!")
                    
                    # Display audio player
                    st.audio(output_path, format='audio/flac')
                    
                    # Provide download button
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Generated Music",
                            data=f,
                            file_name=os.path.basename(output_path),
                            mime='audio/flac'
                        )
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; opacity: 0.7; padding: 1rem;">
    Made with ❤️ using SongBloom | 
    <a href="https://github.com/MASSIVEMAGNETICS/Song-Bloom-Bando-fied-Edition">GitHub</a> | 
    <a href="https://arxiv.org/abs/2506.07634">Paper</a>
</div>
""", unsafe_allow_html=True)

# Initialize session state for first-time users
if 'model' not in st.session_state:
    st.info("👈 Please load the model using the sidebar to get started!")
