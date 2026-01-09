#!/usr/bin/env python3
"""
SongBloom Model Weights Downloader

Downloads model files from Hugging Face Hub to local models directory.
Supports private repos via HUGGING_FACE_HUB_TOKEN environment variable.
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
from typing import Optional


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def verify_checksums(download_dir: Path, repo_id: str) -> bool:
    """
    Verify SHA256 checksums if a checksums file exists in the HF repo.
    Returns True if verification passes or if no checksums file exists.
    """
    try:
        from huggingface_hub import hf_hub_download
        
        # Try to download checksums file
        checksums_path = None
        try:
            cache_dir = download_dir / ".cache"
            checksums_path = hf_hub_download(
                repo_id=repo_id,
                filename="checksums.txt",
                cache_dir=str(cache_dir)
            )
        except Exception:
            # No checksums file, skip verification
            print("ℹ️  No checksums file found in repo, skipping verification")
            return True
        
        # Parse checksums file
        checksums = {}
        with open(checksums_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        checksums[parts[1]] = parts[0]
        
        # Verify each file
        print("🔍 Verifying checksums...")
        all_valid = True
        for filename, expected_hash in checksums.items():
            file_path = download_dir / filename
            if file_path.exists():
                actual_hash = compute_sha256(file_path)
                if actual_hash == expected_hash:
                    print(f"  ✓ {filename}")
                else:
                    print(f"  ✗ {filename} (checksum mismatch)")
                    all_valid = False
            else:
                print(f"  ⚠ {filename} (file not found)")
        
        return all_valid
    except Exception as e:
        print(f"⚠️  Checksum verification failed: {e}")
        return True  # Don't fail if verification has issues


def download_weights(
    repo_id: str = "MASSIVEMAGNETICS/SongBloom-weights",
    target_dir: Optional[str] = None,
    token: Optional[str] = None
) -> int:
    """
    Download model weights from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face model repository ID
        target_dir: Target directory for downloads (default: SongBloom-master/models/songbloom)
        token: HuggingFace token for private repos
    
    Returns:
        0 on success, non-zero on error
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ Error: huggingface_hub package not found")
        print("   Install with: pip install huggingface-hub")
        return 1
    
    # Determine target directory
    if target_dir is None:
        # Default to models/songbloom relative to script location
        script_dir = Path(__file__).parent.parent
        target_dir = script_dir / "models" / "songbloom"
    else:
        target_dir = Path(target_dir)
    
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Get token from environment if not provided
    if token is None:
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    print("=" * 60)
    print("🎵 SongBloom Model Weights Downloader")
    print("=" * 60)
    print(f"Repository: {repo_id}")
    print(f"Target dir: {target_dir}")
    print(f"Token: {'✓ Provided' if token else '✗ Not set (public repos only)'}")
    print()
    
    try:
        print("📥 Downloading model files from Hugging Face Hub...")
        print("   This may take several minutes depending on your connection...")
        print()
        
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            token=token,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        
        print()
        print("✅ Download completed successfully!")
        print(f"   Files saved to: {target_dir}")
        
        # Verify checksums if available
        if not verify_checksums(target_dir, repo_id):
            print()
            print("⚠️  Warning: Some checksums did not match")
            print("   Files may be corrupted. Consider re-downloading.")
            return 1
        
        print()
        print("=" * 60)
        print("✨ Model weights are ready to use!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print()
        print("❌ Error downloading model weights:")
        print(f"   {str(e)}")
        print()
        
        if "401" in str(e) or "403" in str(e) or "authentication" in str(e).lower():
            print("💡 This might be a private repository. To access it:")
            print("   1. Create a token at: https://huggingface.co/settings/tokens")
            print("   2. Set the environment variable:")
            print("      export HUGGING_FACE_HUB_TOKEN=your_token_here")
            print("   3. Run this script again")
        elif "404" in str(e):
            print("💡 Repository not found. Please check:")
            print("   1. The repository ID is correct")
            print("   2. The repository exists on Hugging Face Hub")
            print(f"   3. You have access to: https://huggingface.co/{repo_id}")
        else:
            print("💡 Troubleshooting:")
            print("   - Check your internet connection")
            print("   - Verify the repository exists and is accessible")
            print("   - Try running with --repo-id to specify a different repo")
        
        print()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Download SongBloom model weights from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from default repo to default location
  python download_weights.py
  
  # Download from specific repo
  python download_weights.py --repo-id username/model-name
  
  # Download to custom directory
  python download_weights.py --target-dir /path/to/models
  
  # Use with private repo (token from environment)
  export HUGGING_FACE_HUB_TOKEN=your_token_here
  python download_weights.py --repo-id private/repo
"""
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        default="MASSIVEMAGNETICS/SongBloom-weights",
        help="Hugging Face model repository ID (default: MASSIVEMAGNETICS/SongBloom-weights)"
    )
    
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="Target directory for downloads (default: ../models/songbloom)"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (or use HUGGING_FACE_HUB_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    return download_weights(
        repo_id=args.repo_id,
        target_dir=args.target_dir,
        token=args.token
    )


if __name__ == "__main__":
    sys.exit(main())
