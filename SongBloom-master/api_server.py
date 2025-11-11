"""
SongBloom FastAPI Server
RESTful API for programmatic music generation
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
import torchaudio
import uvicorn
import os
import uuid
from datetime import datetime
from pathlib import Path
import json
import asyncio
from omegaconf import OmegaConf

from SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler

# API Models
class GenerationRequest(BaseModel):
    lyrics: str = Field(..., description="Lyrics for the song")
    num_samples: int = Field(1, ge=1, le=5, description="Number of samples to generate")
    cfg_coef: float = Field(1.5, ge=0.0, le=5.0, description="Classifier-free guidance coefficient")
    steps: int = Field(50, ge=10, le=100, description="Number of diffusion steps")
    top_k: int = Field(200, ge=50, le=500, description="Top-k sampling parameter")


class GenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str
    audio_urls: Optional[List[str]] = None


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float
    message: str
    audio_urls: Optional[List[str]] = None
    created_at: str
    completed_at: Optional[str] = None


# Global state
app = FastAPI(
    title="SongBloom API",
    description="Next-Gen X2 Music Generation API",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
CONFIG = None
JOBS = {}  # Job tracking


def load_model_instance(
    model_name="songbloom_full_150s",
    dtype="bfloat16",
    quantization=None
):
    """Load the SongBloom model"""
    global MODEL, CONFIG
    
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


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("🚀 Starting SongBloom API Server...")
    print("📥 Loading model...")
    
    try:
        load_model_instance()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Failed to load model: {e}")
        print("   The model will need to be loaded manually via /load-model endpoint")


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "SongBloom API",
        "version": "2.0.0",
        "status": "online" if MODEL is not None else "model_not_loaded",
        "endpoints": {
            "generate": "/generate",
            "generate_sync": "/generate/sync",
            "job_status": "/jobs/{job_id}",
            "download": "/download/{job_id}/{sample_id}",
            "health": "/health",
            "models": "/models"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory": {
            "allocated": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "reserved": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        } if torch.cuda.is_available() else None
    }


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": [
            {
                "name": "songbloom_full_150s",
                "size": "2B parameters",
                "max_length": "150s",
                "prompt_type": "10s wav"
            },
            {
                "name": "songbloom_full_150s_dpo",
                "size": "2B parameters",
                "max_length": "150s",
                "prompt_type": "10s wav",
                "note": "DPO post-trained version"
            }
        ],
        "current_model": "songbloom_full_150s" if MODEL is not None else None
    }


@app.post("/load-model")
async def load_model_endpoint(
    model_name: str = "songbloom_full_150s",
    dtype: str = "bfloat16",
    quantization: Optional[str] = None
):
    """Load or reload the model"""
    try:
        load_model_instance(model_name, dtype, quantization)
        return {
            "status": "success",
            "message": f"Model {model_name} loaded successfully with dtype={dtype}, quantization={quantization}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def process_generation(
    job_id: str,
    lyrics: str,
    prompt_audio_path: str,
    num_samples: int,
    cfg_coef: float,
    steps: int,
    top_k: int
):
    """Background task for processing generation"""
    global JOBS
    
    try:
        # Update job status
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["progress"] = 0.1
        
        # Load prompt audio
        prompt_wav, sr = torchaudio.load(prompt_audio_path)
        
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
        output_dir = Path("./api_outputs") / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_samples):
            JOBS[job_id]["progress"] = 0.1 + (0.9 * (i / num_samples))
            
            with torch.cuda.amp.autocast(enabled=True):
                wav = MODEL.generate(lyrics, prompt_wav)
            
            output_path = output_dir / f"sample_{i}.flac"
            torchaudio.save(str(output_path), wav[0].cpu().float(), MODEL.sample_rate)
            output_files.append(f"/download/{job_id}/{i}")
        
        # Update job as completed
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 1.0
        JOBS[job_id]["audio_urls"] = output_files
        JOBS[job_id]["completed_at"] = datetime.now().isoformat()
        JOBS[job_id]["message"] = f"Generated {num_samples} sample(s) successfully"
        
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["message"] = str(e)
        JOBS[job_id]["completed_at"] = datetime.now().isoformat()


@app.post("/generate", response_model=GenerationResponse)
async def generate_async(
    background_tasks: BackgroundTasks,
    lyrics: str = Form(...),
    prompt_audio: UploadFile = File(...),
    num_samples: int = Form(1),
    cfg_coef: float = Form(1.5),
    steps: int = Form(50),
    top_k: int = Form(200)
):
    """Generate music asynchronously"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Create job
    job_id = str(uuid.uuid4())
    
    # Save prompt audio
    prompt_dir = Path("./api_prompts")
    prompt_dir.mkdir(exist_ok=True)
    prompt_path = prompt_dir / f"{job_id}.wav"
    
    with open(prompt_path, "wb") as f:
        f.write(await prompt_audio.read())
    
    # Initialize job
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Job created, waiting to start",
        "audio_urls": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None
    }
    
    # Add to background tasks
    background_tasks.add_task(
        process_generation,
        job_id, lyrics, str(prompt_path),
        num_samples, cfg_coef, steps, top_k
    )
    
    return GenerationResponse(
        job_id=job_id,
        status="pending",
        message="Generation job created successfully. Use /jobs/{job_id} to check status."
    )


@app.post("/generate/sync")
async def generate_sync(
    lyrics: str = Form(...),
    prompt_audio: UploadFile = File(...),
    num_samples: int = Form(1),
    cfg_coef: float = Form(1.5),
    steps: int = Form(50),
    top_k: int = Form(200)
):
    """Generate music synchronously (blocks until complete)"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Create temporary job ID
    job_id = str(uuid.uuid4())
    
    # Save prompt audio
    prompt_dir = Path("./api_prompts")
    prompt_dir.mkdir(exist_ok=True)
    prompt_path = prompt_dir / f"{job_id}.wav"
    
    with open(prompt_path, "wb") as f:
        f.write(await prompt_audio.read())
    
    # Initialize job
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "progress": 0.0,
        "message": "Processing...",
        "audio_urls": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None
    }
    
    # Process synchronously
    await process_generation(
        job_id, lyrics, str(prompt_path),
        num_samples, cfg_coef, steps, top_k
    )
    
    return JOBS[job_id]


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JOBS[job_id]


@app.get("/download/{job_id}/{sample_id}")
async def download_audio(job_id: str, sample_id: int):
    """Download generated audio"""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if JOBS[job_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    file_path = Path("./api_outputs") / job_id / f"sample_{sample_id}.flac"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        file_path,
        media_type="audio/flac",
        filename=f"{job_id}_sample_{sample_id}.flac"
    )


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete job and associated files"""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete output files
    output_dir = Path("./api_outputs") / job_id
    if output_dir.exists():
        for file in output_dir.glob("*"):
            file.unlink()
        output_dir.rmdir()
    
    # Delete prompt file
    prompt_file = Path("./api_prompts") / f"{job_id}.wav"
    if prompt_file.exists():
        prompt_file.unlink()
    
    # Remove from jobs
    del JOBS[job_id]
    
    return {"status": "success", "message": "Job deleted successfully"}


def main():
    """Run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SongBloom API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
