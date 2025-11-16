"""
SongBloom Next-Gen X2 Installation Test
Verify that all components are properly installed
"""
import sys
import importlib
from pathlib import Path


def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        return True, "OK"
    except ImportError as e:
        if package_name:
            return False, f"Missing (install with: pip install {package_name})"
        return False, f"Missing: {str(e)}"


def test_cuda():
    """Test CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return True, f"Available ({device_name})"
        else:
            return False, "Not available (CPU only mode)"
    except Exception as e:
        return False, f"Error: {str(e)}"


def test_flash_attention():
    """Test Flash Attention availability"""
    try:
        import flash_attn
        return True, "Available"
    except ImportError:
        return False, "Not installed (optional, install with: pip install flash-attn)"


def test_file_exists(filepath):
    """Test if a file exists"""
    path = Path(filepath)
    if path.exists():
        return True, f"Found ({path.stat().st_size / 1024:.1f} KB)"
    else:
        return False, "Not found"


def main():
    print("="*70)
    print("SongBloom Next-Gen X2 Installation Test")
    print("="*70)
    print()
    
    # Core dependencies
    print("Core Dependencies:")
    print("-"*70)
    
    tests = [
        ("torch", "torch==2.2.0"),
        ("torchaudio", "torchaudio==2.2.0"),
        ("transformers", "transformers==4.44.1"),
        ("lightning", "lightning==2.2.1"),
        ("omegaconf", "omegaconf==2.2.0"),
        ("einops", "einops==0.8.0"),
        ("huggingface_hub", "huggingface-hub==0.24.6"),
    ]
    
    core_passed = 0
    for module, package in tests:
        success, message = test_import(module, package)
        status = "✓" if success else "✗"
        print(f"  {status} {module:20s} {message}")
        if success:
            core_passed += 1
    
    print()
    
    # Next-Gen X2 dependencies
    print("Next-Gen X2 Dependencies:")
    print("-"*70)
    
    ng_tests = [
        ("gradio", "gradio>=4.0.0"),
        ("fastapi", "fastapi>=0.104.0"),
        ("uvicorn", "uvicorn[standard]>=0.24.0"),
        ("accelerate", "accelerate>=0.24.0"),
        ("bitsandbytes", "bitsandbytes>=0.41.0"),
    ]
    
    ng_passed = 0
    for module, package in ng_tests:
        success, message = test_import(module, package)
        status = "✓" if success else "✗"
        print(f"  {status} {module:20s} {message}")
        if success:
            ng_passed += 1
    
    print()
    
    # Optional dependencies
    print("Optional Dependencies:")
    print("-"*70)
    
    success, message = test_flash_attention()
    status = "✓" if success else "ℹ"
    print(f"  {status} flash_attn         {message}")
    
    print()
    
    # Hardware
    print("Hardware:")
    print("-"*70)
    
    success, message = test_cuda()
    status = "✓" if success else "⚠"
    print(f"  {status} CUDA               {message}")
    
    import torch
    print(f"  ℹ PyTorch version     {torch.__version__}")
    print(f"  ℹ CUDA version        {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    
    print()
    
    # Files
    print("Project Files:")
    print("-"*70)
    
    files = [
        "infer_optimized.py",
        "app.py",
        "api_server.py",
        "benchmark.py",
        "advanced_features.py",
        "export_model.py",
        "NEXTGEN_X2_GUIDE.md",
        "requirements.txt",
    ]
    
    files_passed = 0
    for filepath in files:
        success, message = test_file_exists(filepath)
        status = "✓" if success else "✗"
        print(f"  {status} {filepath:25s} {message}")
        if success:
            files_passed += 1
    
    print()
    print("="*70)
    print("Summary:")
    print("-"*70)
    print(f"Core dependencies:      {core_passed}/{len(tests)} passed")
    print(f"Next-Gen dependencies:  {ng_passed}/{len(ng_tests)} passed")
    print(f"Project files:          {files_passed}/{len(files)} found")
    print()
    
    # Overall status
    if core_passed == len(tests) and files_passed == len(files):
        print("✅ Installation appears to be complete!")
        if ng_passed == len(ng_tests):
            print("✅ All Next-Gen X2 features are available!")
        else:
            print("⚠️  Some Next-Gen X2 features may not be available.")
            print("   Install missing dependencies for full functionality.")
    else:
        print("❌ Installation incomplete. Please install missing dependencies.")
        print("   Run: pip install -r requirements.txt")
    
    print()
    print("Next Steps:")
    print("-"*70)
    if core_passed == len(tests):
        print("1. Try the web interface: python app.py --auto-load-model")
        print("2. Read the guide: cat NEXTGEN_X2_GUIDE.md")
        print("3. Run a benchmark: python benchmark.py --configs fast")
    else:
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Re-run this test: python test_installation.py")
    
    print("="*70)
    
    # Exit code
    if core_passed == len(tests) and files_passed == len(files):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
