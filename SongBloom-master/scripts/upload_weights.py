#!/usr/bin/env python3
"""
SongBloom Model Weights Uploader

Uploads model files from local directory to Hugging Face Hub.
Requires HUGGING_FACE_HUB_TOKEN for authentication.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional


def upload_weights(
    source_dir: str = "weights_to_upload",
    repo_id: str = "MASSIVEMAGNETICS/SongBloom-weights",
    token: Optional[str] = None,
    commit_message: Optional[str] = None
) -> int:
    """
    Upload model weights to Hugging Face Hub.
    
    Args:
        source_dir: Source directory containing files to upload
        repo_id: Hugging Face model repository ID
        token: HuggingFace token for authentication
        commit_message: Custom commit message
    
    Returns:
        0 on success, non-zero on error
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ Error: huggingface_hub package not found")
        print("   Install with: pip install huggingface-hub")
        return 1
    
    # Get token from environment if not provided
    if token is None:
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    if not token:
        print("❌ Error: HUGGING_FACE_HUB_TOKEN is required for uploads")
        print()
        print("💡 To upload weights:")
        print("   1. Create a token at: https://huggingface.co/settings/tokens")
        print("      (Make sure to select 'write' permissions)")
        print("   2. Set the environment variable:")
        print("      export HUGGING_FACE_HUB_TOKEN=your_token_here")
        print("   3. Run this script again")
        print()
        return 1
    
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"❌ Error: Source directory not found: {source_path}")
        print()
        print("💡 Create the directory and add your model files:")
        print(f"   mkdir -p {source_dir}")
        print(f"   cp your_model_files/* {source_dir}/")
        print()
        return 1
    
    # Get list of files to upload
    files_to_upload = []
    for file_path in source_path.rglob("*"):
        if file_path.is_file() and not file_path.name.startswith("."):
            files_to_upload.append(file_path)
    
    if not files_to_upload:
        print(f"❌ Error: No files found in {source_path}")
        print()
        print("💡 Add your model files to the directory:")
        print(f"   cp your_model_files/* {source_dir}/")
        print()
        return 1
    
    print("=" * 60)
    print("🎵 SongBloom Model Weights Uploader")
    print("=" * 60)
    print(f"Repository: {repo_id}")
    print(f"Source dir: {source_path.resolve()}")
    print(f"Files to upload: {len(files_to_upload)}")
    print()
    
    # List files
    print("📦 Files to upload:")
    for file_path in files_to_upload:
        relative_path = file_path.relative_to(source_path)
        file_size = file_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        print(f"   • {relative_path} ({size_mb:.2f} MB)")
    print()
    
    # Confirm upload
    try:
        response = input("Continue with upload? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Upload cancelled.")
            return 0
    except (EOFError, KeyboardInterrupt):
        print("\nUpload cancelled.")
        return 0
    
    print()
    print("📤 Uploading files to Hugging Face Hub...")
    print()
    
    try:
        api = HfApi()
        
        # Verify repo exists or create it
        try:
            api.repo_info(repo_id=repo_id, token=token)
            print(f"✓ Repository {repo_id} exists")
        except Exception:
            print(f"Creating repository {repo_id}...")
            try:
                api.create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
                print(f"✓ Repository created")
            except Exception as e:
                print(f"❌ Failed to create repository: {e}")
                return 1
        
        print()
        
        # Upload each file
        default_message = "Upload model weights"
        message = commit_message if commit_message else default_message
        
        uploaded_count = 0
        for idx, file_path in enumerate(files_to_upload, 1):
            relative_path = file_path.relative_to(source_path)
            path_in_repo = str(relative_path)
            
            print(f"[{idx}/{len(files_to_upload)}] Uploading {relative_path}...")
            
            try:
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    token=token,
                    commit_message=f"{message}: {path_in_repo}"
                )
                print(f"   ✓ Uploaded successfully")
                uploaded_count += 1
            except Exception as e:
                print(f"   ✗ Failed: {e}")
        
        print()
        print("=" * 60)
        
        if uploaded_count == len(files_to_upload):
            print(f"✅ All {uploaded_count} files uploaded successfully!")
            print(f"   View at: https://huggingface.co/{repo_id}")
            print("=" * 60)
            return 0
        else:
            print(f"⚠️  Partial upload: {uploaded_count}/{len(files_to_upload)} files")
            print(f"   Some files failed. Check errors above.")
            print("=" * 60)
            return 1
            
    except Exception as e:
        print()
        print("❌ Error during upload:")
        print(f"   {str(e)}")
        print()
        
        if "401" in str(e) or "403" in str(e) or "authentication" in str(e).lower():
            print("💡 Authentication error. Please check:")
            print("   1. Your token has 'write' permissions")
            print("   2. The token is set correctly:")
            print("      export HUGGING_FACE_HUB_TOKEN=your_token_here")
            print("   3. You have access to the repository")
        else:
            print("💡 Troubleshooting:")
            print("   - Check your internet connection")
            print("   - Verify the repository ID is correct")
            print("   - Ensure you have write access to the repository")
        
        print()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Upload SongBloom model weights to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload from default directory to default repo
  export HUGGING_FACE_HUB_TOKEN=your_token_here
  python upload_weights.py
  
  # Upload from custom directory
  python upload_weights.py --source-dir /path/to/weights
  
  # Upload to specific repo
  python upload_weights.py --repo-id username/model-name
  
  # With custom commit message
  python upload_weights.py --commit-message "Add fine-tuned weights v2"

Note:
  - Requires HUGGING_FACE_HUB_TOKEN environment variable
  - Token must have 'write' permissions
  - Create token at: https://huggingface.co/settings/tokens
"""
    )
    
    parser.add_argument(
        "--source-dir",
        type=str,
        default="weights_to_upload",
        help="Source directory containing files to upload (default: weights_to_upload)"
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        default="MASSIVEMAGNETICS/SongBloom-weights",
        help="Hugging Face model repository ID (default: MASSIVEMAGNETICS/SongBloom-weights)"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (or use HUGGING_FACE_HUB_TOKEN env var)"
    )
    
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Custom commit message for uploads"
    )
    
    args = parser.parse_args()
    
    return upload_weights(
        source_dir=args.source_dir,
        repo_id=args.repo_id,
        token=args.token,
        commit_message=args.commit_message
    )


if __name__ == "__main__":
    sys.exit(main())
