"""
SongBloom Performance Benchmarking Script
Test different optimization configurations
"""
import torch
import torchaudio
import time
import json
import argparse
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

from infer_optimized import load_config, hf_download, optimize_model, apply_quantization
from SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler


def benchmark_configuration(
    config_name,
    model_name="songbloom_full_150s",
    dtype="bfloat16",
    quantization=None,
    optimization_level="standard",
    lyrics="Verse 1:\nIn the morning light, I see your face",
    prompt_audio_path=None,
    num_runs=3
):
    """Benchmark a specific configuration"""
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}")
    
    results = {
        'config_name': config_name,
        'model_name': model_name,
        'dtype': dtype,
        'quantization': quantization,
        'optimization_level': optimization_level,
        'num_runs': num_runs,
        'runs': []
    }
    
    try:
        # Load model
        print("Loading model...")
        start_load = time.time()
        
        local_dir = "./cache"
        hf_download(model_name, local_dir)
        cfg = load_config(f"{local_dir}/{model_name}.yaml", parent_dir=local_dir)
        cfg.max_dur = cfg.max_dur + 20
        
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16
        }
        dtype_torch = dtype_map[dtype]
        
        model = SongBloom_Sampler.build_from_trainer(cfg, strict=True, dtype=dtype_torch)
        
        if optimization_level != 'minimal':
            model.diffusion = optimize_model(model.diffusion, optimization_level)
        
        if quantization:
            model.diffusion = apply_quantization(model.diffusion, quantization)
        
        gen_params = cfg.inference if hasattr(cfg, 'inference') else {
            'cfg_coef': 1.5,
            'steps': 50,
            'dit_cfg_type': 'h',
            'use_sampling': True,
            'top_k': 200,
            'max_frames': cfg.max_dur * 25
        }
        model.set_generation_params(**gen_params)
        
        load_time = time.time() - start_load
        print(f"Model loaded in {load_time:.2f}s")
        
        # Memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1e9
            print(f"Initial VRAM: {initial_memory:.2f} GB")
        
        # Load prompt audio
        if prompt_audio_path and Path(prompt_audio_path).exists():
            prompt_wav, sr = torchaudio.load(prompt_audio_path)
            if sr != model.sample_rate:
                prompt_wav = torchaudio.functional.resample(prompt_wav, sr, model.sample_rate)
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True).to(dtype_torch)
            prompt_wav = prompt_wav[..., :10*model.sample_rate]
        else:
            # Create dummy prompt
            prompt_wav = torch.randn(1, 10*model.sample_rate).to(dtype_torch)
        
        # Warmup run
        print("Warmup run...")
        with torch.cuda.amp.autocast(enabled=(dtype != 'float32')):
            _ = model.generate(lyrics, prompt_wav)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark runs
        generation_times = []
        for i in range(num_runs):
            print(f"Run {i+1}/{num_runs}...")
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            start_gen = time.time()
            with torch.cuda.amp.autocast(enabled=(dtype != 'float32')):
                wav = model.generate(lyrics, prompt_wav)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            gen_time = time.time() - start_gen
            generation_times.append(gen_time)
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1e9
            else:
                peak_memory = 0
            
            run_result = {
                'run_id': i + 1,
                'generation_time': gen_time,
                'peak_vram_gb': peak_memory,
                'audio_length': wav.shape[-1] / model.sample_rate
            }
            results['runs'].append(run_result)
            
            print(f"  Time: {gen_time:.2f}s, Peak VRAM: {peak_memory:.2f} GB")
        
        # Compute statistics
        results['stats'] = {
            'load_time': load_time,
            'mean_generation_time': np.mean(generation_times),
            'std_generation_time': np.std(generation_times),
            'min_generation_time': np.min(generation_times),
            'max_generation_time': np.max(generation_times),
            'mean_peak_vram_gb': np.mean([r['peak_vram_gb'] for r in results['runs']]),
            'audio_length': results['runs'][0]['audio_length']
        }
        
        print(f"\nResults:")
        print(f"  Mean generation time: {results['stats']['mean_generation_time']:.2f}s ± {results['stats']['std_generation_time']:.2f}s")
        print(f"  Mean peak VRAM: {results['stats']['mean_peak_vram_gb']:.2f} GB")
        print(f"  Audio length: {results['stats']['audio_length']:.2f}s")
        
        results['status'] = 'success'
        
    except Exception as e:
        print(f"Error: {e}")
        results['status'] = 'failed'
        results['error'] = str(e)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="SongBloom Performance Benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="Output JSON file for results")
    parser.add_argument("--prompt-audio", type=str, default=None,
                       help="Path to prompt audio file")
    parser.add_argument("--num-runs", type=int, default=3,
                       help="Number of runs per configuration")
    parser.add_argument("--configs", type=str, nargs='+', default=['all'],
                       help="Configurations to test (all, fast, quality, memory)")
    
    args = parser.parse_args()
    
    # Define benchmark configurations
    all_configs = {
        'baseline_fp32': {
            'dtype': 'float32',
            'quantization': None,
            'optimization_level': 'minimal'
        },
        'standard_bf16': {
            'dtype': 'bfloat16',
            'quantization': None,
            'optimization_level': 'standard'
        },
        'optimized_bf16': {
            'dtype': 'bfloat16',
            'quantization': None,
            'optimization_level': 'aggressive'
        },
        'memory_int8': {
            'dtype': 'bfloat16',
            'quantization': 'int8',
            'optimization_level': 'standard'
        },
        'ultra_memory_int8': {
            'dtype': 'bfloat16',
            'quantization': 'int8',
            'optimization_level': 'aggressive'
        }
    }
    
    # Select configurations to run
    if 'all' in args.configs:
        configs_to_run = all_configs
    elif 'fast' in args.configs:
        configs_to_run = {
            'optimized_bf16': all_configs['optimized_bf16'],
            'ultra_memory_int8': all_configs['ultra_memory_int8']
        }
    elif 'quality' in args.configs:
        configs_to_run = {
            'baseline_fp32': all_configs['baseline_fp32'],
            'standard_bf16': all_configs['standard_bf16']
        }
    elif 'memory' in args.configs:
        configs_to_run = {
            'memory_int8': all_configs['memory_int8'],
            'ultra_memory_int8': all_configs['ultra_memory_int8']
        }
    else:
        # Custom selection
        configs_to_run = {k: all_configs[k] for k in args.configs if k in all_configs}
    
    print("="*60)
    print("SongBloom Next-Gen X2 Performance Benchmark")
    print("="*60)
    print(f"\nConfigurations to test: {list(configs_to_run.keys())}")
    print(f"Runs per configuration: {args.num_runs}")
    
    # Run benchmarks
    all_results = []
    for config_name, config_params in configs_to_run.items():
        result = benchmark_configuration(
            config_name=config_name,
            prompt_audio_path=args.prompt_audio,
            num_runs=args.num_runs,
            **config_params
        )
        all_results.append(result)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results
    output_data = {
        'benchmark_info': {
            'num_runs': args.num_runs,
            'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__
        },
        'results': all_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Benchmark complete! Results saved to {args.output}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nSummary:")
    print(f"{'Configuration':<25} {'Time (s)':<12} {'VRAM (GB)':<12} {'Status':<10}")
    print("-"*60)
    for result in all_results:
        if result['status'] == 'success':
            time_str = f"{result['stats']['mean_generation_time']:.2f}"
            vram_str = f"{result['stats']['mean_peak_vram_gb']:.2f}"
            status = "✓"
        else:
            time_str = "-"
            vram_str = "-"
            status = "✗"
        print(f"{result['config_name']:<25} {time_str:<12} {vram_str:<12} {status:<10}")


if __name__ == "__main__":
    main()
