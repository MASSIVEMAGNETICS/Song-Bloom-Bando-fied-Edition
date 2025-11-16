

<p align="center"><img src="docs/icon.png" width="50%"></p>


# **SongBloom**: *Coherent Song Generation via Interleaved Autoregressive Sketching and Diffusion Refinement*

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-2506.07634-b31b1b.svg)](https://arxiv.org/abs/2506.07634)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/CypressYang/SongBloom)
[![Demo Page](https://img.shields.io/badge/Demo-Audio%20Samples-green)](https://cypress-yang.github.io/SongBloom_demo)

</div>

We propose **SongBloom**, a novel framework for full-length song generation that leverages an interleaved paradigm of autoregressive sketching and diffusion-based refinement. SongBloom employs an autoregressive diffusion model that combines the high fidelity of diffusion models with the scalability of language models.
Specifically, it gradually extends a musical sketch from short to long and refines the details from coarse to fine-grained. The interleaved generation paradigm effectively integrates prior semantic and acoustic context to guide the generation process.
Experimental results demonstrate that SongBloom outperforms existing methods across both subjective and objective metrics and achieves performance comparable to the state-of-the-art commercial music generation platforms.

![img](docs/architecture.png)



## Models

| Name                 | Size | Max Length | Prompt type | 🤗                                            |
| -------------------- | ---- | ---------- | ----------- | -------------------------------------------- |
| songbloom_full_150s  | 2B   | 2m30s      | 10s wav     | [link](https://huggingface.co/CypressYang/SongBloom) |
| songbloom_full_150s_dpo  | 2B   | 2m30s      | 10s wav     | [link](https://huggingface.co/CypressYang/SongBloom) |
| songbloom_mulan_150s | 2B   | 2m30s      | 10s wav / text description |           coming soon                           |
| ... |      |            |             |                                              |


## Updates
- **Nov 2025**: 🚀 **Next-Gen X2 Upgrade** - Major release with optimizations, GUI, and API
- **Sep 2025**: Release the songbloom_full_150s model with DPO post-training
- **Jun 2025**: Release the songbloom_full_150s and inference script

## ✨ Next-Gen X2 Features

This repository includes the **Next-Gen X2 upgrade** with significant enhancements:

### 🎯 Core Improvements
- **⚡ Advanced Optimizations**: INT8/INT4 quantization, Flash Attention 2, mixed precision
- **🎨 Modern Web GUI**: Suno-like interface with Gradio for interactive generation
- **🔌 RESTful API**: FastAPI server for programmatic access
- **💾 Memory Efficient**: Run on GPUs with as little as 2GB VRAM
- **🚀 Performance**: 2-4x faster inference with optimizations

### 🎮 Quick Start

**Option 1: Interactive Web Interface (Recommended)**
```bash
./quickstart.sh
# Choose option 1 for GUI
```

**Option 2: Optimized Command-Line**
```bash
python infer_optimized.py \
  --input-jsonl example/test.jsonl \
  --dtype bfloat16 \
  --quantization int8
```

**Option 3: API Server**
```bash
python api_server.py
# Access docs at http://localhost:8000/docs
```

📖 **For detailed documentation, see [NEXTGEN_X2_GUIDE.md](NEXTGEN_X2_GUIDE.md)**



## Getting Started

### Prepare Environments

```bash
conda create -n SongBloom python==3.8.12
conda activate SongBloom

# yum install libsndfile
# pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118 # For different CUDA version
pip install -r requirements.txt
```

### Data Preparation

A  .jsonl file, where each line is a json object:

```json
{
	"idx": "The index of each sample", 
	"lyrics": "The lyrics to be generated",
	"prompt_wav": "The path of the style prompt audio",
}
```

One example can be refered to as: [example/test.jsonl](example/test.jsonl)

The prompt wav should be a 10-second, 48kHz audio clip.

For details on lyric formatting, see [docs/lyric_format.md](docs/lyric_format.md).

### Inference

```bash
source set_env.sh

python3 infer.py --input-jsonl example/test.jsonl


# For GPUs with low VRAM like RTX4090, you should set the dtype as bfloat16
python3 infer.py --input-jsonl example/test.jsonl --dtype bfloat16

# SongBloom also supports flash-attn (optional). To enable it, please install flash-attn (v2.6.3 is used during training) manually and set os.environ['DISABLE_FLASH_ATTN'] = "0" in infer.py:8
```

- model-name: Specify model version, see the model cards (eg: songbloom_full_150s/songbloom_full_150s_dpo);
- local-dir: Dir where the weights and config files are downloaded;
- input-jsonl: input raw data;
- output-dir: Dir where the output audio saved;
- n-samples: How many audios will be generated for each input term;

## Citation

```
@article{yang2025songbloom,
title={SongBloom: Coherent Song Generation via Interleaved Autoregressive Sketching and Diffusion Refinement},
author={Yang, Chenyu and Wang, Shuai and Chen, Hangting and Tan, Wei and Yu, Jianwei and Li, Haizhou},
journal={arXiv preprint arXiv:2506.07634},
year={2025}
}
```

## License

SongBloom (codes and weights) is released under the [LICENSE](LICENSE). 
