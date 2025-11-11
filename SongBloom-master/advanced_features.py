"""
SongBloom Advanced Features
Style mixing, continuation, and experimental capabilities
"""
import torch
import torchaudio
import numpy as np
from typing import List, Optional
import argparse
from pathlib import Path

from infer_optimized import load_config, hf_download, optimize_model
from SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler


def mix_prompts(prompts: List[torch.Tensor], weights: Optional[List[float]] = None):
    """
    Mix multiple style prompts with optional weights
    
    Args:
        prompts: List of prompt audio tensors
        weights: Optional weights for each prompt (default: equal weights)
    
    Returns:
        Mixed prompt tensor
    """
    if weights is None:
        weights = [1.0 / len(prompts)] * len(prompts)
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
    
    # Ensure all prompts have same shape
    min_length = min(p.shape[-1] for p in prompts)
    prompts = [p[..., :min_length] for p in prompts]
    
    # Mix
    mixed = sum(w * p for w, p in zip(weights, prompts))
    
    return mixed


def interpolate_prompts(prompt1: torch.Tensor, prompt2: torch.Tensor, alpha: float = 0.5):
    """
    Interpolate between two style prompts
    
    Args:
        prompt1: First prompt
        prompt2: Second prompt
        alpha: Interpolation factor (0 = prompt1, 1 = prompt2)
    
    Returns:
        Interpolated prompt
    """
    # Ensure same length
    min_length = min(prompt1.shape[-1], prompt2.shape[-1])
    prompt1 = prompt1[..., :min_length]
    prompt2 = prompt2[..., :min_length]
    
    # Linear interpolation
    interpolated = (1 - alpha) * prompt1 + alpha * prompt2
    
    return interpolated


def generate_variations(
    model: SongBloom_Sampler,
    lyrics: str,
    prompt_audio: torch.Tensor,
    num_variations: int = 3,
    temperature_range: tuple = (0.8, 1.2),
    cfg_range: tuple = (1.0, 2.0)
):
    """
    Generate multiple variations of the same song with different parameters
    
    Args:
        model: SongBloom model instance
        lyrics: Song lyrics
        prompt_audio: Style prompt
        num_variations: Number of variations to generate
        temperature_range: Range for temperature variation
        cfg_range: Range for CFG coefficient variation
    
    Returns:
        List of generated audio tensors
    """
    variations = []
    
    # Generate temperature and CFG values
    temps = np.linspace(temperature_range[0], temperature_range[1], num_variations)
    cfgs = np.linspace(cfg_range[0], cfg_range[1], num_variations)
    
    for i, (temp, cfg) in enumerate(zip(temps, cfgs)):
        print(f"Generating variation {i+1}/{num_variations} (temp={temp:.2f}, cfg={cfg:.2f})...")
        
        # Update generation parameters
        model.set_generation_params(
            cfg_coef=cfg,
            steps=50,
            dit_cfg_type='h',
            use_sampling=True,
            top_k=200,
            max_frames=model.max_duration * 25
        )
        
        # Generate
        with torch.cuda.amp.autocast(enabled=True):
            wav = model.generate(lyrics, prompt_audio)
        
        variations.append(wav)
    
    return variations


def extend_audio(
    model: SongBloom_Sampler,
    existing_audio: torch.Tensor,
    continuation_lyrics: str,
    overlap_duration: float = 2.0
):
    """
    Extend existing audio with new generation
    
    Args:
        model: SongBloom model instance
        existing_audio: Audio to extend
        continuation_lyrics: Lyrics for continuation
        overlap_duration: Overlap duration in seconds for smooth transition
    
    Returns:
        Extended audio tensor
    """
    # Use end of existing audio as prompt
    prompt_duration = 10  # seconds
    prompt_samples = prompt_duration * model.sample_rate
    
    if existing_audio.shape[-1] < prompt_samples:
        # Audio too short, use all of it
        prompt = existing_audio
    else:
        # Use last 10 seconds as prompt
        prompt = existing_audio[..., -prompt_samples:]
    
    # Generate continuation
    print("Generating continuation...")
    with torch.cuda.amp.autocast(enabled=True):
        continuation = model.generate(continuation_lyrics, prompt)
    
    # Crossfade overlap region
    overlap_samples = int(overlap_duration * model.sample_rate)
    
    if overlap_samples > 0 and existing_audio.shape[-1] > overlap_samples:
        # Create crossfade
        fade_out = torch.linspace(1, 0, overlap_samples).to(existing_audio.device)
        fade_in = torch.linspace(0, 1, overlap_samples).to(continuation.device)
        
        # Apply fades
        existing_end = existing_audio[..., -overlap_samples:] * fade_out
        continuation_start = continuation[..., :overlap_samples] * fade_in
        
        # Combine
        crossfade = existing_end + continuation_start
        
        # Concatenate
        extended = torch.cat([
            existing_audio[..., :-overlap_samples],
            crossfade,
            continuation[..., overlap_samples:]
        ], dim=-1)
    else:
        # No crossfade, simple concatenation
        extended = torch.cat([existing_audio, continuation], dim=-1)
    
    return extended


def style_transfer(
    model: SongBloom_Sampler,
    lyrics: str,
    style_prompts: List[torch.Tensor],
    style_weights: Optional[List[float]] = None
):
    """
    Generate song with mixed style from multiple prompts
    
    Args:
        model: SongBloom model instance
        lyrics: Song lyrics
        style_prompts: List of style prompt audios
        style_weights: Weights for each style
    
    Returns:
        Generated audio with mixed style
    """
    # Mix prompts
    mixed_prompt = mix_prompts(style_prompts, style_weights)
    
    # Generate with mixed style
    print("Generating with mixed style...")
    with torch.cuda.amp.autocast(enabled=True):
        wav = model.generate(lyrics, mixed_prompt)
    
    return wav


def main():
    parser = argparse.ArgumentParser(description="SongBloom Advanced Features")
    parser.add_argument("--model-name", type=str, default="songbloom_full_150s")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Style mixing
    mix_parser = subparsers.add_parser('mix', help='Mix multiple style prompts')
    mix_parser.add_argument("--lyrics", type=str, required=True)
    mix_parser.add_argument("--prompts", type=str, nargs='+', required=True)
    mix_parser.add_argument("--weights", type=float, nargs='+', default=None)
    mix_parser.add_argument("--output", type=str, default="mixed_output.flac")
    
    # Variations
    var_parser = subparsers.add_parser('variations', help='Generate variations')
    var_parser.add_argument("--lyrics", type=str, required=True)
    var_parser.add_argument("--prompt", type=str, required=True)
    var_parser.add_argument("--num-variations", type=int, default=3)
    var_parser.add_argument("--output-dir", type=str, default="./variations")
    
    # Extension
    ext_parser = subparsers.add_parser('extend', help='Extend existing audio')
    ext_parser.add_argument("--input-audio", type=str, required=True)
    ext_parser.add_argument("--continuation-lyrics", type=str, required=True)
    ext_parser.add_argument("--output", type=str, default="extended_output.flac")
    ext_parser.add_argument("--overlap", type=float, default=2.0)
    
    # Interpolation
    interp_parser = subparsers.add_parser('interpolate', help='Interpolate between styles')
    interp_parser.add_argument("--lyrics", type=str, required=True)
    interp_parser.add_argument("--prompt1", type=str, required=True)
    interp_parser.add_argument("--prompt2", type=str, required=True)
    interp_parser.add_argument("--steps", type=int, default=5)
    interp_parser.add_argument("--output-dir", type=str, default="./interpolation")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Load model
    print("Loading model...")
    local_dir = "./cache"
    hf_download(args.model_name, local_dir)
    
    cfg = load_config(f"{local_dir}/{args.model_name}.yaml", parent_dir=local_dir)
    cfg.max_dur = cfg.max_dur + 20
    
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    model = SongBloom_Sampler.build_from_trainer(cfg, strict=True, dtype=dtype)
    model.diffusion = optimize_model(model.diffusion, 'standard')
    
    gen_params = cfg.inference if hasattr(cfg, 'inference') else {
        'cfg_coef': 1.5,
        'steps': 50,
        'dit_cfg_type': 'h',
        'use_sampling': True,
        'top_k': 200,
        'max_frames': cfg.max_dur * 25
    }
    model.set_generation_params(**gen_params)
    
    print("✓ Model loaded\n")
    
    # Execute command
    if args.command == 'mix':
        # Load prompts
        prompts = []
        for prompt_path in args.prompts:
            wav, sr = torchaudio.load(prompt_path)
            if sr != model.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
            wav = wav.mean(dim=0, keepdim=True).to(dtype)
            wav = wav[..., :10*model.sample_rate]
            prompts.append(wav)
        
        # Generate with mixed style
        result = style_transfer(model, args.lyrics, prompts, args.weights)
        
        # Save
        torchaudio.save(args.output, result[0].cpu().float(), model.sample_rate)
        print(f"✓ Saved to: {args.output}")
    
    elif args.command == 'variations':
        # Load prompt
        wav, sr = torchaudio.load(args.prompt)
        if sr != model.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
        wav = wav.mean(dim=0, keepdim=True).to(dtype)
        wav = wav[..., :10*model.sample_rate]
        
        # Generate variations
        variations = generate_variations(
            model, args.lyrics, wav, args.num_variations
        )
        
        # Save
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        for i, var in enumerate(variations):
            output_path = output_dir / f"variation_{i+1}.flac"
            torchaudio.save(str(output_path), var[0].cpu().float(), model.sample_rate)
            print(f"✓ Saved variation {i+1} to: {output_path}")
    
    elif args.command == 'extend':
        # Load existing audio
        existing, sr = torchaudio.load(args.input_audio)
        if sr != model.sample_rate:
            existing = torchaudio.functional.resample(existing, sr, model.sample_rate)
        existing = existing.mean(dim=0, keepdim=True).to(dtype)
        
        # Extend
        result = extend_audio(
            model, existing, args.continuation_lyrics, args.overlap
        )
        
        # Save
        torchaudio.save(args.output, result[0].cpu().float(), model.sample_rate)
        print(f"✓ Saved to: {args.output}")
    
    elif args.command == 'interpolate':
        # Load prompts
        wav1, sr = torchaudio.load(args.prompt1)
        if sr != model.sample_rate:
            wav1 = torchaudio.functional.resample(wav1, sr, model.sample_rate)
        wav1 = wav1.mean(dim=0, keepdim=True).to(dtype)
        wav1 = wav1[..., :10*model.sample_rate]
        
        wav2, sr = torchaudio.load(args.prompt2)
        if sr != model.sample_rate:
            wav2 = torchaudio.functional.resample(wav2, sr, model.sample_rate)
        wav2 = wav2.mean(dim=0, keepdim=True).to(dtype)
        wav2 = wav2[..., :10*model.sample_rate]
        
        # Generate interpolations
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        alphas = np.linspace(0, 1, args.steps)
        
        for i, alpha in enumerate(alphas):
            print(f"Generating interpolation {i+1}/{args.steps} (alpha={alpha:.2f})...")
            
            # Interpolate prompts
            prompt = interpolate_prompts(wav1, wav2, alpha)
            
            # Generate
            with torch.cuda.amp.autocast(enabled=True):
                result = model.generate(args.lyrics, prompt)
            
            # Save
            output_path = output_dir / f"interpolation_{i+1:02d}.flac"
            torchaudio.save(str(output_path), result[0].cpu().float(), model.sample_rate)
            print(f"✓ Saved to: {output_path}")


if __name__ == "__main__":
    main()
