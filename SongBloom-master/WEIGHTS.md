# SongBloom Model Weights Management

This document describes how to manage SongBloom model weights using Hugging Face Hub for storage and distribution.

## Overview

SongBloom uses Hugging Face Hub to store and distribute model weights separately from the code repository. This approach:
- Keeps the Git repository lightweight
- Enables version control for large model files
- Allows both public and private weight distribution
- Supports automated downloads during setup

## Quick Start

### Downloading Model Weights

#### Option 1: Automatic Download (Recommended)

The quickstart script will automatically check for weights and prompt you to download them:

```bash
./quickstart.sh
```

#### Option 2: Manual Download

Download weights from the default public repository:

```bash
cd SongBloom-master
python scripts/download_weights.py
```

For private repositories, set your Hugging Face token first:

```bash
export HUGGING_FACE_HUB_TOKEN=your_token_here
python scripts/download_weights.py --repo-id username/private-repo
```

#### Option 3: Docker Runtime

Use the runtime Dockerfile that downloads weights on container start:

```bash
cd SongBloom-master
docker build -f Dockerfile.runtime -t songbloom:runtime .
docker run -p 7860:7860 -e HUGGING_FACE_HUB_TOKEN=your_token songbloom:runtime
```

## For Model Creators and Maintainers

### Setting Up a Hugging Face Model Repository

1. **Create a Hugging Face Account**
   - Sign up at: https://huggingface.co/join

2. **Create a Model Repository**
   - Go to: https://huggingface.co/new
   - Choose a repository name (e.g., `YourUsername/SongBloom-weights`)
   - Set visibility (Public or Private)
   - Click "Create model"

3. **Create an Access Token**
   - Go to: https://huggingface.co/settings/tokens
   - Click "New token"
   - Name: `SongBloom-upload`
   - Role: Select "Write"
   - Click "Generate a token"
   - **Save this token securely!**

### Uploading Model Weights

#### Method 1: Local Upload (For Initial Setup)

1. **Prepare your weights directory:**

```bash
mkdir -p weights_to_upload
cp path/to/your/model/files/* weights_to_upload/
```

2. **Upload to Hugging Face:**

```bash
export HUGGING_FACE_HUB_TOKEN=your_write_token_here
cd SongBloom-master
python scripts/upload_weights.py --repo-id YourUsername/SongBloom-weights
```

The script will:
- Show you the list of files to upload
- Ask for confirmation
- Upload each file with progress tracking
- Report success or any errors

#### Method 2: CI/CD Upload (For Automated Updates)

1. **Add GitHub Secret:**
   - Go to your repository settings
   - Navigate to: Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `HUGGING_FACE_HUB_TOKEN`
   - Value: Your write token
   - Click "Add secret"

2. **Prepare weights in repository:**

```bash
# In your local repository
mkdir -p weights_to_upload
cp path/to/your/model/files/* weights_to_upload/
git add weights_to_upload/
git commit -m "Add weights for upload"
git push
```

3. **Run the GitHub Action:**
   - Go to: Actions → "Push Model Weights to Hugging Face"
   - Click "Run workflow"
   - Configure:
     - **repo_id**: Your Hugging Face repo (e.g., `YourUsername/SongBloom-weights`)
     - **weights_path**: Path to weights in repo (default: `weights_to_upload`)
     - **commit_message**: Description of the upload
   - Click "Run workflow"

4. **Verify upload:**
   - Check the Action logs for success
   - Visit your Hugging Face repo to confirm files are uploaded
   - Test download: `python scripts/download_weights.py --repo-id YourUsername/SongBloom-weights`

## Advanced Usage

### Custom Download Location

Download weights to a specific directory:

```bash
python scripts/download_weights.py --target-dir /path/to/custom/location
```

### Checksum Verification

The download script automatically verifies checksums if a `checksums.txt` file exists in your Hugging Face repository.

To create a checksums file for your repository:

```bash
cd weights_to_upload
sha256sum * > checksums.txt
```

Upload this file along with your model weights.

### Using Different Repositories

#### Download from a specific repo:

```bash
python scripts/download_weights.py --repo-id organization/model-name
```

#### Upload to a specific repo:

```bash
python scripts/upload_weights.py \
  --source-dir /path/to/weights \
  --repo-id organization/model-name \
  --commit-message "Add version 2.0 weights"
```

## Private Repository Access

For private model repositories:

1. **Create a Read Token:**
   - Go to: https://huggingface.co/settings/tokens
   - Create a token with "Read" permissions

2. **Set Environment Variable:**

```bash
export HUGGING_FACE_HUB_TOKEN=your_read_token
```

3. **Download:**

```bash
python scripts/download_weights.py --repo-id username/private-repo
```

## Docker Deployment

### Build Runtime Image

```bash
cd SongBloom-master
docker build -f Dockerfile.runtime -t songbloom:runtime .
```

### Run with Public Weights

```bash
docker run -p 7860:7860 songbloom:runtime
```

Weights will be downloaded automatically on first startup.

### Run with Private Weights

```bash
docker run -p 7860:7860 \
  -e HUGGING_FACE_HUB_TOKEN=your_token \
  songbloom:runtime
```

### Mount Pre-downloaded Weights

```bash
docker run -p 7860:7860 \
  -v /path/to/local/weights:/app/models/songbloom:ro \
  songbloom:runtime
```

## Troubleshooting

### Download Issues

**Problem:** "401 Unauthorized" or "403 Forbidden"
- **Solution:** The repository is private. Set `HUGGING_FACE_HUB_TOKEN` with a valid token that has access.

**Problem:** "404 Not Found"
- **Solution:** Check the repository ID is correct and exists on Hugging Face Hub.

**Problem:** "Checksum mismatch"
- **Solution:** Re-download the weights. If the problem persists, the uploaded files may be corrupted.

### Upload Issues

**Problem:** "HUGGING_FACE_HUB_TOKEN is required"
- **Solution:** Set the token with write permissions: `export HUGGING_FACE_HUB_TOKEN=your_write_token`

**Problem:** "Permission denied"
- **Solution:** Ensure your token has "Write" permissions and you have access to the repository.

**Problem:** "Repository not found"
- **Solution:** Create the repository on Hugging Face Hub first, or the script will attempt to create it automatically.

## Best Practices

1. **Version Your Weights:**
   - Use clear version numbers in commit messages
   - Tag releases on Hugging Face Hub
   - Document changes in model cards

2. **Test Before Uploading:**
   - Verify model weights work locally
   - Test the download script after uploading
   - Check file sizes and checksums

3. **Secure Your Tokens:**
   - Never commit tokens to Git
   - Use GitHub Secrets for CI/CD
   - Rotate tokens periodically
   - Use read-only tokens when possible

4. **Document Your Models:**
   - Add a README.md to your Hugging Face repo
   - Include model architecture details
   - Document training procedures and datasets
   - Provide usage examples

## Default Repository

The default repository for SongBloom weights is:
- **Repository ID:** `MASSIVEMAGNETICS/SongBloom-weights`
- **URL:** https://huggingface.co/MASSIVEMAGNETICS/SongBloom-weights

To use a different repository, specify `--repo-id` when running the scripts.

## Support

For issues related to:
- **Script errors:** Check the error messages and this documentation
- **Hugging Face Hub:** Visit https://huggingface.co/docs
- **SongBloom model:** Check the main README.md and documentation

## License

Model weights may have different licensing than the code. Check the Hugging Face repository for specific license information.
