#!/usr/bin/env python3
"""
Cognitive Architecture Example
Demonstrates the revolutionary Level 2 capabilities with fractal memory and concept algebra.
"""
import sys
from pathlib import Path

# Add SongBloom to path
sys.path.insert(0, str(Path(__file__).parent / "SongBloom-master"))

import torch
from SongBloom.models.fractal_memory import FractalMemory, HyperdimensionalVector
from datetime import datetime, timedelta


def example_fractal_memory():
    """Demonstrate fractal memory with recursive compression"""
    print("=" * 80)
    print("EXAMPLE 1: Fractal Memory - Recursive Compression")
    print("=" * 80)
    
    # Initialize fractal memory
    memory = FractalMemory(hd_dimension=10000)
    
    # Simulate storing daily music generations over 2 weeks
    print("\n📝 Storing daily music generations...")
    
    base_date = datetime(2025, 1, 1)
    
    daily_entries = [
        ("A funky jazz tune with saxophone. Very upbeat and energetic.", {"genre": "jazz", "mood": "upbeat"}),
        ("A melancholic piano ballad about lost love.", {"genre": "ballad", "mood": "sad"}),
        ("Electronic dance music with heavy bass drops.", {"genre": "edm", "mood": "energetic"}),
        ("Classical symphony with strings and brass.", {"genre": "classical", "mood": "majestic"}),
        ("Acoustic guitar folk song about nature.", {"genre": "folk", "mood": "peaceful"}),
        ("Heavy metal with distorted guitars.", {"genre": "metal", "mood": "intense"}),
        ("Smooth R&B with soulful vocals.", {"genre": "rnb", "mood": "smooth"}),
        ("Country song with banjo and harmonica.", {"genre": "country", "mood": "nostalgic"}),
        ("Jazz fusion with experimental rhythms.", {"genre": "jazz-fusion", "mood": "complex"}),
        ("Ambient electronic soundscape.", {"genre": "ambient", "mood": "dreamy"}),
        ("Blues with slide guitar.", {"genre": "blues", "mood": "melancholic"}),
        ("Pop song with catchy hooks.", {"genre": "pop", "mood": "happy"}),
        ("Reggae with island vibes.", {"genre": "reggae", "mood": "relaxed"}),
        ("Hip-hop with boom bap beats.", {"genre": "hiphop", "mood": "groovy"}),
    ]
    
    for i, (content, metadata) in enumerate(daily_entries):
        date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        memory.store_daily_memory(date, content, metadata)
        print(f"  ✓ Day {i+1}: {content[:50]}...")
    
    # Show statistics
    print("\n📊 Memory Statistics:")
    stats = memory.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Query the memory
    print("\n🔍 Querying Memory:")
    queries = [
        "jazz music",
        "sad piano",
        "energetic dance music",
        "peaceful nature sounds"
    ]
    
    for query in queries:
        print(f"\n  Query: '{query}'")
        results = memory.query_memory(query, top_k=3)
        for rank, result in enumerate(results, 1):
            mem = result['memory']
            print(f"    {rank}. [{result['level']}] {result['id']} - Similarity: {result['similarity']:.3f}")
            if 'content' in mem:
                print(f"       Content: {mem['content'][:60]}...")
    
    # Save memory to disk
    print("\n💾 Saving fractal memory to disk...")
    memory.save_to_disk()
    print("  ✓ Saved successfully!")
    
    return memory


def example_concept_algebra():
    """Demonstrate concept algebra with hyperdimensional vectors"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Concept Algebra - Hyperdimensional Computing")
    print("=" * 80)
    
    # Initialize hyperdimensional vector system
    hdv = HyperdimensionalVector(dimension=10000)
    
    print("\n🧮 Creating concept vectors...")
    
    # Create concept vectors
    concepts = {
        'Apple': hdv.create_random_vector(),
        'Red': hdv.create_random_vector(),
        'Gravity': hdv.create_random_vector(),
        'Newton': hdv.create_random_vector(),
        'Jazz': hdv.create_random_vector(),
        'Saxophone': hdv.create_random_vector(),
        'Upbeat': hdv.create_random_vector(),
        'Funky': hdv.create_random_vector(),
    }
    
    print("  Created concepts:", list(concepts.keys()))
    
    # Demonstrate concept algebra
    print("\n🔬 Concept Algebra Operations:")
    
    # Example 1: Apple * Red + Gravity ≈ Newton
    print("\n  Example 1: Physics Concept")
    print("    Vector(Apple) * Vector(Red) + Vector(Gravity) ≈ ?")
    
    result = hdv.concept_algebra(concepts, "Apple * Red + Gravity")
    similarity_newton = hdv.similarity(result, concepts['Newton'])
    
    print(f"    Similarity to Newton: {similarity_newton:.3f}")
    print("    (Note: Random initialization means low similarity, but concept is demonstrated)")
    
    # Example 2: Jazz * Saxophone + Upbeat ≈ Funky Jazz
    print("\n  Example 2: Music Concept")
    print("    Vector(Jazz) * Vector(Saxophone) + Vector(Upbeat) ≈ ?")
    
    result = hdv.concept_algebra(concepts, "Jazz * Saxophone + Upbeat")
    similarity_funky = hdv.similarity(result, concepts['Funky'])
    
    print(f"    Similarity to Funky: {similarity_funky:.3f}")
    
    # Demonstrate binding and bundling
    print("\n  🔗 Binding (relationship/association):")
    bound = hdv.bind(concepts['Jazz'], concepts['Saxophone'])
    print(f"    Jazz * Saxophone creates a bound concept vector")
    
    print("\n  📦 Bundling (superposition/union):")
    bundled = hdv.bundle([concepts['Jazz'], concepts['Saxophone'], concepts['Upbeat']])
    print(f"    Jazz + Saxophone + Upbeat creates a bundled concept vector")
    
    # Demonstrate permutation (sequence)
    print("\n  🔄 Permutation (sequence/order):")
    permuted = hdv.permute_vector(concepts['Jazz'], shift=1)
    similarity_before = hdv.similarity(concepts['Jazz'], permuted)
    print(f"    Original vs Permuted similarity: {similarity_before:.3f}")
    print(f"    (Lower similarity shows permutation changes the vector)")
    
    return concepts


def example_model_selection():
    """Demonstrate intelligent model selection"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Intelligent Model Selection")
    print("=" * 80)
    
    try:
        from SongBloom.models.model_selector import ModelSelector, ModelRegistry, CognitiveLevel
        
        # Initialize selector
        selector = ModelSelector()
        
        print("\n🧠 Available Models:")
        models = ModelRegistry.list_models()
        for model_type, info in models.items():
            print(f"\n  {model_type}:")
            print(f"    Level: {info['cognitive_level'].value}")
            print(f"    Description: {info['description']}")
        
        print("\n🎯 Intelligent Selection:")
        
        # Task-based selection
        tasks = [
            ("music_generation", None),
            ("music_generation", CognitiveLevel.LEVEL_2_HOLOGRAPHIC),
            ("text_to_music", CognitiveLevel.LEVEL_2_HOLOGRAPHIC),
        ]
        
        for task, level in tasks:
            selected = selector.select_model(task, cognitive_level=level)
            level_str = f" (Level: {level.value})" if level else ""
            print(f"  Task: {task}{level_str}")
            print(f"    → Selected: {selected}")
        
    except Exception as e:
        print(f"\n⚠️ Model selection example requires SongBloom models: {e}")


def main():
    """Run all examples"""
    print("\n🚀 Cognitive Architecture Examples")
    print("Demonstrating Level 2: Holographic & Hyperdimensional Computing\n")
    
    # Example 1: Fractal Memory
    memory = example_fractal_memory()
    
    # Example 2: Concept Algebra
    concepts = example_concept_algebra()
    
    # Example 3: Model Selection
    example_model_selection()
    
    print("\n" + "=" * 80)
    print("✨ Examples Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Fractal Memory: Hierarchical compression (Day → Week → Month → Year)")
    print("  2. Concept Algebra: Mathematical operations on abstract concepts")
    print("  3. Holographic Properties: Distributed, robust representation")
    print("  4. Intelligent Selection: Task-aware model selection")
    print("\nNext Steps:")
    print("  - Try the Streamlit interface: streamlit run streamlit_app.py")
    print("  - Read COGNITIVE_ARCHITECTURE.md for detailed documentation")
    print("  - Experiment with your own concept algebras")
    print()


if __name__ == "__main__":
    main()
