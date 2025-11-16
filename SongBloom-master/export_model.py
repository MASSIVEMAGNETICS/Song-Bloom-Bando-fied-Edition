"""
SongBloom Model Export Utilities
Export models to different formats for deployment
"""
import torch
import argparse
from pathlib import Path
import json

from infer_optimized import load_config, hf_download
from SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler


def export_to_torchscript(model, output_path, example_inputs=None):
    """
    Export model to TorchScript format
    
    Args:
        model: SongBloom model instance
        output_path: Path to save the exported model
        example_inputs: Example inputs for tracing
    """
    print("Exporting to TorchScript...")
    
    # Set model to eval mode
    model.diffusion.eval()
    
    try:
        # Script the model (better than trace for control flow)
        scripted_model = torch.jit.script(model.diffusion)
        
        # Save
        scripted_model.save(output_path)
        print(f"✓ Saved TorchScript model to: {output_path}")
        
        # Test loading
        loaded = torch.jit.load(output_path)
        print(f"✓ Verified: Model loads successfully")
        
        return True
    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")
        print("  Note: Not all models are compatible with TorchScript")
        return False


def export_to_onnx(model, output_path, example_inputs):
    """
    Export model to ONNX format
    
    Args:
        model: SongBloom model instance
        output_path: Path to save the exported model
        example_inputs: Example inputs for tracing
    """
    print("Exporting to ONNX...")
    
    model.diffusion.eval()
    
    try:
        # Export to ONNX
        torch.onnx.export(
            model.diffusion,
            example_inputs,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'sequence_length'},
                'output': {0: 'batch_size', 2: 'sequence_length'}
            }
        )
        
        print(f"✓ Saved ONNX model to: {output_path}")
        
        # Verify with onnx
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print(f"✓ Verified: ONNX model is valid")
        except ImportError:
            print("  Note: Install onnx package to verify exported model")
        
        return True
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False


def export_state_dict(model, output_path, half_precision=False):
    """
    Export model state dict with optional half precision
    
    Args:
        model: SongBloom model instance
        output_path: Path to save the state dict
        half_precision: Convert to half precision (FP16)
    """
    print(f"Exporting state dict (half_precision={half_precision})...")
    
    state_dict = model.diffusion.state_dict()
    
    if half_precision:
        # Convert to half precision
        state_dict = {k: v.half() if v.dtype == torch.float32 else v 
                     for k, v in state_dict.items()}
    
    # Save
    torch.save({
        'model_state_dict': state_dict,
        'config': model.cfg if hasattr(model, 'cfg') else None,
        'half_precision': half_precision
    }, output_path)
    
    print(f"✓ Saved state dict to: {output_path}")
    
    # Report size
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"  File size: {size_mb:.2f} MB")
    
    return True


def quantize_model_dynamic(model, output_path):
    """
    Apply dynamic quantization and save
    
    Args:
        model: SongBloom model instance
        output_path: Path to save the quantized model
    """
    print("Applying dynamic quantization...")
    
    try:
        # Quantize
        quantized_model = torch.quantization.quantize_dynamic(
            model.diffusion,
            {torch.nn.Linear, torch.nn.Conv1d},
            dtype=torch.qint8
        )
        
        # Save
        torch.save({
            'model': quantized_model,
            'quantized': True
        }, output_path)
        
        print(f"✓ Saved quantized model to: {output_path}")
        
        # Report size
        size_mb = Path(output_path).stat().st_size / 1e6
        print(f"  File size: {size_mb:.2f} MB")
        
        return True
    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        return False


def export_model_info(model, output_path):
    """
    Export model architecture and parameter information
    
    Args:
        model: SongBloom model instance
        output_path: Path to save the info JSON
    """
    print("Exporting model information...")
    
    info = {
        'model_type': 'SongBloom',
        'architecture': str(type(model.diffusion).__name__),
        'parameters': {
            'total': sum(p.numel() for p in model.diffusion.parameters()),
            'trainable': sum(p.numel() for p in model.diffusion.parameters() if p.requires_grad)
        },
        'sample_rate': model.sample_rate,
        'max_duration': model.max_duration,
        'generation_params': model.generation_params
    }
    
    # Add layer information
    layers = []
    for name, module in model.diffusion.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            num_params = sum(p.numel() for p in module.parameters())
            if num_params > 0:
                layers.append({
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': num_params
                })
    
    info['layers'] = layers[:20]  # First 20 layers
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✓ Saved model info to: {output_path}")
    print(f"  Total parameters: {info['parameters']['total']:,}")
    print(f"  Trainable parameters: {info['parameters']['trainable']:,}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Export SongBloom models")
    parser.add_argument("--model-name", type=str, default="songbloom_full_150s",
                       help="Model name to export")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument("--output-dir", type=str, default="./exports",
                       help="Output directory for exported models")
    
    # Export formats
    parser.add_argument("--export-torchscript", action="store_true",
                       help="Export to TorchScript")
    parser.add_argument("--export-onnx", action="store_true",
                       help="Export to ONNX (experimental)")
    parser.add_argument("--export-state-dict", action="store_true",
                       help="Export state dict")
    parser.add_argument("--export-quantized", action="store_true",
                       help="Export quantized model")
    parser.add_argument("--export-info", action="store_true",
                       help="Export model information")
    parser.add_argument("--export-all", action="store_true",
                       help="Export all formats")
    
    # Options
    parser.add_argument("--half-precision", action="store_true",
                       help="Use half precision for state dict")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("SongBloom Model Export Utility")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
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
    print("✓ Model loaded")
    
    # Export formats
    results = {}
    
    if args.export_all or args.export_info:
        output_path = output_dir / f"{args.model_name}_info.json"
        results['info'] = export_model_info(model, output_path)
    
    if args.export_all or args.export_state_dict:
        suffix = "_fp16" if args.half_precision else ""
        output_path = output_dir / f"{args.model_name}_state_dict{suffix}.pt"
        results['state_dict'] = export_state_dict(
            model, output_path, args.half_precision
        )
    
    if args.export_all or args.export_quantized:
        output_path = output_dir / f"{args.model_name}_quantized.pt"
        results['quantized'] = quantize_model_dynamic(model, output_path)
    
    if args.export_all or args.export_torchscript:
        output_path = output_dir / f"{args.model_name}_torchscript.pt"
        results['torchscript'] = export_to_torchscript(model, output_path)
    
    # ONNX export requires example inputs
    if args.export_all or args.export_onnx:
        print("\nNote: ONNX export requires example inputs and may not work with complex models")
        # Skip ONNX for now as it requires proper input tracing
        results['onnx'] = False
    
    # Summary
    print("\n" + "="*60)
    print("Export Summary")
    print("="*60)
    for format_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {format_name}")
    
    print(f"\nExported models saved to: {output_dir}")


if __name__ == "__main__":
    main()
