# SongBloom Streamlit Deployment Guide

This directory contains everything needed to deploy SongBloom as a Streamlit app.

## Files

- **streamlit_app.py** - Main Streamlit application
- **requirements.txt** - Python dependencies
- **packages.txt** - System dependencies (ffmpeg, libsndfile1)
- **.streamlit/config.toml** - Streamlit configuration

## Deployment

### Streamlit Cloud (Recommended for Demo)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Click "New app"
4. Select your repository: `MASSIVEMAGNETICS/Song-Bloom-Bando-fied-Edition`
5. Set main file path: `streamlit_app.py`
6. Click "Deploy"

**⚠️ Important Notes:**
- Streamlit Cloud's free tier has limited CPU/RAM resources
- SongBloom requires significant computational resources (ideally GPU)
- For production use, deploy on GPU-enabled infrastructure
- The app may timeout or fail on Streamlit Cloud's free tier
- Consider using Streamlit Cloud for Teams with GPU support or deploy elsewhere

### Local Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

Access at: http://localhost:8501

### Production Deployment

For production deployments with GPU support, consider:

1. **AWS EC2 with GPU** (p2/p3/g4 instances)
   ```bash
   # Install CUDA and dependencies
   pip install -r requirements.txt
   streamlit run streamlit_app.py --server.port 80 --server.address 0.0.0.0
   ```

2. **Google Cloud Platform with GPU**
   - Use Compute Engine with GPU
   - Follow similar setup as AWS

3. **Azure with GPU**
   - Use NC-series VMs
   - Follow similar setup as AWS

4. **Docker Deployment**
   ```bash
   # Build and run with GPU support
   docker build -t songbloom-streamlit .
   docker run --gpus all -p 8501:8501 songbloom-streamlit
   ```

## Usage

1. **Load Model** - Click "Load Model" in the sidebar (first run downloads ~2GB model)
2. **Enter Lyrics** - Provide lyrics for your song
3. **Upload Style Audio** - Upload a 10-second audio clip for style reference
4. **Adjust Settings** - Optional: modify CFG, steps, top-k parameters
5. **Generate** - Click "Generate Music" and wait for generation (2-5 minutes with GPU)
6. **Download** - Download your generated song

## System Requirements

- **Minimum (CPU-only):** 8GB RAM, will be very slow
- **Recommended:** NVIDIA GPU with 6GB+ VRAM, 16GB RAM
- **Optimal:** NVIDIA GPU with 12GB+ VRAM, 32GB RAM

## Troubleshooting

### Model Loading Fails
- Check internet connection (downloads from HuggingFace)
- Ensure sufficient disk space (~5GB)
- Try reducing precision to float16 or bfloat16

### Generation Too Slow
- Use GPU instead of CPU
- Reduce diffusion steps (30-50 recommended)
- Use quantization (int8 recommended for GPU with <12GB VRAM)

### Out of Memory
- Use smaller precision (float16 or bfloat16)
- Enable quantization (int8 or int4)
- Reduce diffusion steps

## Support

For issues and questions:
- GitHub Issues: [Song-Bloom-Bando-fied-Edition/issues](https://github.com/MASSIVEMAGNETICS/Song-Bloom-Bando-fied-Edition/issues)
- Original SongBloom: [Paper](https://arxiv.org/abs/2506.07634)
