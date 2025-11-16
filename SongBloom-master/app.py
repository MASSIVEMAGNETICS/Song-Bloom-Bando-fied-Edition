"""
SongBloom Gradio Web Interface - Suno-like GUI
Modern web interface for interactive music generation
"""
import os
import sys
import json
import torch
import torchaudio
import gradio as gr
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
from datetime import datetime

# Import SongBloom components
from SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler

# Global model instance
MODEL = None
CONFIG = None


def load_model(model_name="songbloom_full_150s", dtype="float32", quantization=None):
    """Load the SongBloom model"""
    global MODEL, CONFIG
    
    try:
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
        dtype_torch = dtype_map[dtype]
        
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
        
        MODEL = model
        CONFIG = cfg
        
        return "✅ Model loaded successfully!"
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"


def generate_music(
    lyrics,
    prompt_audio,
    num_samples=1,
    cfg_coef=1.5,
    steps=50,
    top_k=200,
    progress=gr.Progress()
):
    """Generate music from lyrics and prompt audio"""
    global MODEL
    
    if MODEL is None:
        return None, "❌ Please load the model first!"
    
    if not lyrics or lyrics.strip() == "":
        return None, "❌ Please provide lyrics!"
    
    if prompt_audio is None:
        return None, "❌ Please provide a prompt audio file!"
    
    try:
        progress(0, desc="Loading prompt audio...")
        
        # Load prompt audio
        if isinstance(prompt_audio, str):
            prompt_wav, sr = torchaudio.load(prompt_audio)
        else:
            # Handle tuple from Gradio (sr, data)
            sr, prompt_data = prompt_audio
            prompt_wav = torch.from_numpy(prompt_data).float()
            if len(prompt_wav.shape) == 1:
                prompt_wav = prompt_wav.unsqueeze(0)
            else:
                prompt_wav = prompt_wav.T
        
        # Resample if needed
        if sr != MODEL.sample_rate:
            prompt_wav = torchaudio.functional.resample(prompt_wav, sr, MODEL.sample_rate)
        
        # Convert to mono and truncate
        prompt_wav = prompt_wav.mean(dim=0, keepdim=True)
        prompt_wav = prompt_wav[..., :10*MODEL.sample_rate]
        
        # Update generation parameters
        MODEL.set_generation_params(
            cfg_coef=cfg_coef,
            steps=steps,
            top_k=top_k,
            dit_cfg_type='h',
            use_sampling=True,
            max_frames=CONFIG.max_dur * 25
        )
        
        # Generate samples
        output_files = []
        for i in range(num_samples):
            progress((i+1)/num_samples, desc=f"Generating sample {i+1}/{num_samples}...")
            
            with torch.cuda.amp.autocast(enabled=True):
                wav = MODEL.generate(lyrics, prompt_wav)
            
            # Save output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("./gradio_outputs")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"generated_{timestamp}_s{i}.flac"
            
            torchaudio.save(str(output_path), wav[0].cpu().float(), MODEL.sample_rate)
            output_files.append(str(output_path))
        
        # Return first generated file and success message
        message = f"✅ Generated {num_samples} sample(s) successfully!"
        return output_files[0], message
        
    except Exception as e:
        return None, f"❌ Error during generation: {str(e)}"


def create_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for Suno-like styling
    custom_css = """
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        max-width: 1200px;
        margin: auto;
    }
    .header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .generate-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.8rem 2rem !important;
    }
    .settings-box {
        background: #f7f7f7;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as interface:
        
        # Header
        gr.HTML("""
            <div class="header">
                <h1>🎵 SongBloom Next-Gen X2</h1>
                <p>AI-Powered Song Generation with Advanced Optimizations</p>
            </div>
        """)
        
        with gr.Tabs():
            # Main Generation Tab
            with gr.Tab("🎼 Generate Music"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input")
                        
                        lyrics_input = gr.TextArea(
                            label="Lyrics",
                            placeholder="Enter your lyrics here...\n\nExample:\nVerse 1:\nIn the morning light...",
                            lines=10,
                            max_lines=20
                        )
                        
                        prompt_audio_input = gr.Audio(
                            label="Style Prompt Audio (10 seconds)",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            num_samples = gr.Slider(
                                minimum=1, maximum=5, value=1, step=1,
                                label="Number of Samples"
                            )
                            cfg_coef = gr.Slider(
                                minimum=0.0, maximum=5.0, value=1.5, step=0.1,
                                label="Guidance Coefficient (CFG)"
                            )
                            steps = gr.Slider(
                                minimum=10, maximum=100, value=50, step=10,
                                label="Diffusion Steps"
                            )
                            top_k = gr.Slider(
                                minimum=50, maximum=500, value=200, step=50,
                                label="Top-K Sampling"
                            )
                        
                        generate_btn = gr.Button(
                            "🎵 Generate Music",
                            variant="primary",
                            elem_classes="generate-btn"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Output")
                        
                        output_audio = gr.Audio(
                            label="Generated Music",
                            type="filepath"
                        )
                        
                        output_message = gr.Textbox(
                            label="Status",
                            lines=2,
                            interactive=False
                        )
                        
                        gr.Markdown("""
                        ### 💡 Tips
                        - Upload a 10-second audio clip that represents the style you want
                        - Write clear, structured lyrics with verses, chorus, etc.
                        - Higher CFG values (2-3) give more creative results
                        - More diffusion steps (50-100) improve quality but take longer
                        - Generate multiple samples to find the best result
                        """)
                
                # Connect generation
                generate_btn.click(
                    fn=generate_music,
                    inputs=[
                        lyrics_input,
                        prompt_audio_input,
                        num_samples,
                        cfg_coef,
                        steps,
                        top_k
                    ],
                    outputs=[output_audio, output_message]
                )
            
            # Model Settings Tab
            with gr.Tab("⚙️ Model Settings"):
                gr.Markdown("### Model Configuration")
                
                with gr.Row():
                    model_name = gr.Dropdown(
                        choices=["songbloom_full_150s", "songbloom_full_150s_dpo"],
                        value="songbloom_full_150s",
                        label="Model Version"
                    )
                    
                    dtype = gr.Dropdown(
                        choices=["float32", "float16", "bfloat16"],
                        value="bfloat16",
                        label="Precision"
                    )
                    
                    quantization = gr.Dropdown(
                        choices=["None", "int8", "int4"],
                        value="None",
                        label="Quantization"
                    )
                
                load_model_btn = gr.Button("Load Model", variant="primary")
                model_status = gr.Textbox(
                    label="Model Status",
                    value="⚠️ Model not loaded. Click 'Load Model' to start.",
                    interactive=False
                )
                
                load_model_btn.click(
                    fn=lambda m, d, q: load_model(m, d, None if q == "None" else q),
                    inputs=[model_name, dtype, quantization],
                    outputs=model_status
                )
                
                gr.Markdown("""
                ### Optimization Guide
                
                **Precision:**
                - `float32`: Highest quality, most memory
                - `bfloat16`: Good balance (recommended for RTX 30/40 series)
                - `float16`: Lowest memory, may have numerical issues
                
                **Quantization:**
                - `None`: No quantization (default)
                - `int8`: ~2x memory reduction, minimal quality loss
                - `int4`: ~4x memory reduction, some quality loss (requires bitsandbytes)
                
                **Memory Requirements:**
                - float32: ~8GB VRAM
                - bfloat16: ~4GB VRAM
                - bfloat16 + int8: ~2GB VRAM
                """)
            
            # Presets Tab
            with gr.Tab("📚 Presets & Examples"):
                gr.Markdown("### Example Prompts")
                
                examples = gr.Examples(
                    examples=[
                        [
                            "Verse 1:\nIn the morning light, I see your face\nMemories dancing in this sacred space\n\nChorus:\nWe're alive, we're alive tonight\nUnderneath the stars so bright",
                            None,
                            2,
                            1.5,
                            50,
                            200
                        ],
                        [
                            "Verse 1:\nCity lights are calling me home\nThrough the streets where I used to roam\n\nChorus:\nTake me back to where we started\nBefore we were broken hearted",
                            None,
                            1,
                            2.0,
                            50,
                            200
                        ]
                    ],
                    inputs=[
                        lyrics_input,
                        prompt_audio_input,
                        num_samples,
                        cfg_coef,
                        steps,
                        top_k
                    ],
                    label="Try these examples (you still need to provide prompt audio)"
                )
            
            # About Tab
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                # SongBloom Next-Gen X2
                
                ## About
                SongBloom is a state-of-the-art AI music generation system that creates coherent,
                full-length songs from lyrics and style prompts.
                
                ## Features
                - ✨ Full song generation (up to 2.5 minutes)
                - 🎨 Style transfer from audio prompts
                - 🚀 Optimized inference with quantization
                - 💾 Memory-efficient processing
                - 🎯 High-quality audio output (48kHz)
                
                ## Next-Gen X2 Enhancements
                - ⚡ Flash Attention 2 support
                - 🗜️ INT8/INT4 quantization
                - 🔧 Mixed precision training
                - 📊 Advanced generation controls
                - 🎛️ Modern web interface
                
                ## Citation
                ```
                @article{yang2025songbloom,
                  title={SongBloom: Coherent Song Generation via Interleaved Autoregressive Sketching and Diffusion Refinement},
                  author={Yang, Chenyu and Wang, Shuai and Chen, Hangting and Tan, Wei and Yu, Jianwei and Li, Haizhou},
                  journal={arXiv preprint arXiv:2506.07634},
                  year={2025}
                }
                ```
                
                ## Links
                - 📄 [Paper](https://arxiv.org/abs/2506.07634)
                - 🤗 [Models](https://huggingface.co/CypressYang/SongBloom)
                - 🎧 [Demo](https://cypress-yang.github.io/SongBloom_demo)
                - 💻 [GitHub](https://github.com/MASSIVEMAGNETICS/Song-Bloom-Bando-fied-Edition)
                """)
    
    return interface


def main():
    """Launch the Gradio interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SongBloom Web Interface")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port")
    parser.add_argument("--auto-load-model", action="store_true", help="Automatically load model on startup")
    
    args = parser.parse_args()
    
    # Auto-load model if requested
    if args.auto_load_model:
        print("Auto-loading model...")
        result = load_model()
        print(result)
    
    # Create and launch interface
    interface = create_interface()
    interface.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True
    )


if __name__ == "__main__":
    main()
