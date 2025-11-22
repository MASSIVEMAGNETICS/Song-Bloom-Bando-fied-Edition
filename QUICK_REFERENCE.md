# Cognitive Architecture - Quick Reference Guide

## 🚀 Quick Start

### Option 1: Run the Demo
```bash
python example_cognitive_architecture.py
```

### Option 2: Use Streamlit UI
```bash
streamlit run streamlit_app.py
```

### Option 3: Python API
```python
from SongBloom.models.model_selector import ModelSelector, CognitiveLevel
from SongBloom.models.fractal_memory import FractalMemory

# Select and load Level 2 model
selector = ModelSelector()
model = selector.load_model(model_type="diffusion_transformer")

# Use fractal memory
memory = FractalMemory()
memory.store_daily_memory("2025-01-15", "Generated funky jazz")
results = memory.query_memory("jazz music", top_k=5)
```

## 📚 Key Concepts

### The Four Levels

| Level | Name | Status | Description |
|-------|------|--------|-------------|
| 1 | Foundation | ✅ Existing | Standard RAG with vector databases |
| 2 | Holographic | ✅ **Implemented** | Hyperdimensional computing with concept algebra |
| 3 | Active Inference | 🔶 Foundation | Predictive memory that anticipates needs |
| 4 | Neuromorphic | 🔷 Roadmap | Quantum & memristor-based hardware |

### Concept Algebra

**Mathematical operations on abstract concepts:**

```python
# Bind (*): Relationship/Association
Apple * Red = "Red Apple"

# Bundle (+): Superposition/Union  
Apple + Fruit + Healthy = "Healthy Fruit"

# Example: Derive Newton from components
Vector(Apple) * Vector(Red) + Vector(Gravity) ≈ Vector(Newton)
```

### Fractal Memory

**Hierarchical time-scale organization:**

```
Daily Memories (7) → Weekly Summary (1)
Weekly Summaries (4) → Monthly Summary (1)
Monthly Summaries (12) → Yearly Summary (1)
```

## 🎯 Common Tasks

### 1. Generate Music with Level 2 Model

**Streamlit UI:**
1. Open sidebar
2. Select "Level 2: Holographic"
3. Choose "diffusion_transformer"
4. Load model
5. Enter lyrics and generate

**Python:**
```python
from SongBloom.models.diffusion_transformer import MusicDiffusionTransformer

model = MusicDiffusionTransformer(dim=512, num_layers=6, num_heads=8)
output = model.generate("A funky jazz tune", duration_sec=10.0, steps=50)
```

### 2. Use Fractal Memory

**Store:**
```python
from SongBloom.models.fractal_memory import FractalMemory

memory = FractalMemory(hd_dimension=10000)
memory.store_daily_memory(
    "2025-01-15",
    "Generated a funky jazz tune with saxophone",
    metadata={"genre": "jazz", "mood": "upbeat"}
)
```

**Query:**
```python
results = memory.query_memory("jazz music", top_k=5)
for result in results:
    print(f"Level: {result['level']}, Similarity: {result['similarity']:.3f}")
```

**Save/Load:**
```python
# Save to disk
memory.save_to_disk()

# Load from disk
memory.load_from_disk()

# Statistics
stats = memory.get_statistics()
print(stats)
```

### 3. Perform Concept Algebra

```python
from SongBloom.models.fractal_memory import HyperdimensionalVector

hdv = HyperdimensionalVector(dimension=10000)

# Create concepts
concepts = {
    'Jazz': hdv.create_random_vector(),
    'Saxophone': hdv.create_random_vector(),
    'Upbeat': hdv.create_random_vector()
}

# Combine concepts
result = hdv.concept_algebra(concepts, "Jazz * Saxophone + Upbeat")

# Check similarity
similarity = hdv.similarity(result, concepts['Jazz'])
```

### 4. Intelligent Model Selection

```python
from SongBloom.models.model_selector import ModelSelector, CognitiveLevel

selector = ModelSelector()

# Auto-select based on task
model_type = selector.select_model(
    task="music_generation",
    cognitive_level=CognitiveLevel.LEVEL_2_HOLOGRAPHIC,
    requirements={"max_memory": "8GB"}
)

# Load the selected model
model = selector.load_model(model_type)
```

## 📖 Documentation

- **[COGNITIVE_ARCHITECTURE.md](COGNITIVE_ARCHITECTURE.md)** - Complete guide (12,000+ chars)
- **[COGNITIVE_ARCHITECTURE_DIAGRAMS.md](COGNITIVE_ARCHITECTURE_DIAGRAMS.md)** - Visual diagrams
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical details
- **[README.md](README.md)** - Main documentation

## 🔧 Configuration

### Model Configuration

**Level 1 (Original):**
```python
config = {
    'model': 'songbloom_original',
    'dtype': 'float32',
    'quantization': None
}
```

**Level 2 (Holographic):**
```python
config = {
    'model': 'diffusion_transformer',
    'dim': 512,
    'num_layers': 6,
    'num_heads': 8,
    'mel_channels': 80
}
```

### Memory Configuration

```python
memory_config = {
    'hd_dimension': 10000,  # Hyperdimensional vector size
    'save_path': './fractal_memory'  # Storage location
}
```

## ⚡ Performance Tips

### Speed vs Quality

| Setting | Speed | Quality | Use Case |
|---------|-------|---------|----------|
| L1, FP16, 30 steps | 2x faster | Good | Quick iterations |
| L2, FP32, 50 steps | 1x (baseline) | Excellent | Production |
| L2, FP32, 100 steps | 0.5x slower | Outstanding | Final masters |

### Memory Optimization

1. **Reduce HD Dimension:**
   ```python
   memory = FractalMemory(hd_dimension=5000)  # Less memory, faster
   ```

2. **Use Quantization:**
   ```python
   model = load_model(quantization='int8')  # 50% less VRAM
   ```

3. **Limit Query Depth:**
   ```python
   results = memory.query_memory(query, level='daily', top_k=3)
   ```

## 🐛 Troubleshooting

### Common Issues

**Issue:** Model loading fails
```python
# Solution: Check if model is registered
from SongBloom.models.model_selector import ModelRegistry
models = ModelRegistry.list_models()
print(models)
```

**Issue:** Fractal memory not saving
```python
# Solution: Check directory permissions
memory = FractalMemory(save_path='./my_memory')
memory.save_to_disk()
```

**Issue:** Low similarity scores in concept algebra
```
# This is expected! Random initialization means low baseline similarity.
# In production, you would:
# 1. Train concept vectors from actual data
# 2. Use pre-trained embeddings
# 3. Fine-tune on domain-specific concepts
```

## 🧪 Testing

### Run Examples
```bash
# Basic example
python example_cognitive_architecture.py

# With environment variable (skip auto-init)
SONGBLOOM_SKIP_AUTO_INIT=1 python example_cognitive_architecture.py
```

### Verify Installation
```bash
# Check syntax
python -m py_compile SongBloom-master/SongBloom/models/diffusion_transformer.py

# Run Streamlit
streamlit run streamlit_app.py
```

## 💡 Tips & Tricks

### 1. Optimize for Your Use Case

**Fast Prototyping:**
- Use Level 1 with quantization
- Reduce diffusion steps (30)
- Lower HD dimension (5000)

**Production Quality:**
- Use Level 2 full precision
- Increase diffusion steps (100)
- Full HD dimension (10000)

### 2. Build Domain Knowledge

```python
# Store domain-specific memories over time
for day in range(30):
    memory.store_daily_memory(
        f"2025-01-{day+1:02d}",
        generate_music_description(),
        metadata={"domain": "jazz"}
    )

# Automatic compression builds hierarchical knowledge
```

### 3. Experiment with Concept Algebra

```python
# Create a music concept library
music_concepts = {
    'Jazz': hdv.create_random_vector(),
    'Blues': hdv.create_random_vector(),
    'Funk': hdv.create_random_vector(),
    'Smooth': hdv.create_random_vector(),
    'Energetic': hdv.create_random_vector()
}

# Explore combinations
fusion = hdv.concept_algebra(music_concepts, "Jazz * Funk + Energetic")
smooth_jazz = hdv.concept_algebra(music_concepts, "Jazz * Smooth")
```

## 🚀 Next Steps

1. **Try the Demo:**
   ```bash
   python example_cognitive_architecture.py
   ```

2. **Explore the Docs:**
   - Read COGNITIVE_ARCHITECTURE.md
   - Review visual diagrams
   - Study code examples

3. **Build Something:**
   - Create your own concept library
   - Build a music recommendation system
   - Experiment with fractal memory

4. **Contribute:**
   - Share your findings
   - Improve documentation
   - Add new features

## 📞 Getting Help

- **Documentation:** Read the comprehensive guides
- **Examples:** Run example_cognitive_architecture.py
- **Issues:** Check GitHub issues
- **Community:** Join discussions

## 🎓 Learn More

**Scientific Background:**
- Hyperdimensional Computing: Kanerva (2009)
- Active Inference: Friston (2010)
- Diffusion Models: Ho et al. (2020)
- DiT Architecture: Peebles & Xie (2023)

**Resources:**
- TorchHD library: github.com/hyperdimensional-computing/torchhd
- Free Energy Principle papers
- Neuromorphic computing research

---

**🌟 Remember:** This isn't just a model upgrade—it's a paradigm shift from passive retrieval to active cognitive architecture!
