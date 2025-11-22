"""
Fractal Memory System - Level 2 Cognitive Architecture
Implements Holographic & Hyperdimensional Computing for music generation.

This is the "revolutionary" Level 2 approach that treats memory as fractal and holographic,
allowing recursive summarization and hyperdimensional concept algebra.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Union
import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HyperdimensionalVector(nn.Module):
    """
    Hyperdimensional vector representation that allows concept algebra.
    Based on Holographic & Hyperdimensional Computing principles.
    """
    
    def __init__(self, dimension: int = 10000):
        """
        Initialize hyperdimensional vector space.
        
        Args:
            dimension: Dimensionality of hypervectors (typically 10,000+)
        """
        super().__init__()
        self.dimension = dimension
        
    def create_random_vector(self) -> torch.Tensor:
        """Create a random hypervector"""
        # Bipolar random vector {-1, +1}
        return torch.sign(torch.randn(self.dimension))
    
    def bind(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Bind two hypervectors (represents relationship/binding).
        In HDC, binding is often element-wise multiplication.
        """
        return v1 * v2
    
    def bundle(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """
        Bundle multiple hypervectors (represents superposition/union).
        In HDC, bundling is element-wise addition followed by thresholding.
        """
        bundled = torch.stack(vectors).sum(dim=0)
        return torch.sign(bundled)
    
    def permute_vector(self, v: torch.Tensor, shift: int = 1) -> torch.Tensor:
        """
        Permute hypervector (represents sequence/order).
        Circular rotation of vector elements.
        """
        return torch.roll(v, shifts=shift, dims=0)
    
    def similarity(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        """
        Compute similarity between two hypervectors.
        Uses cosine similarity normalized to [-1, 1].
        """
        return float(torch.dot(v1, v2) / self.dimension)
    
    def concept_algebra(self, concepts: Dict[str, torch.Tensor], expression: str) -> torch.Tensor:
        """
        Perform concept algebra on hypervectors.
        
        Example:
            Vector(Apple) * Vector(Red) + Vector(Gravity) ≈ Vector(Newton)
        
        Args:
            concepts: Dictionary mapping concept names to hypervectors
            expression: Algebraic expression (e.g., "Apple * Red + Gravity")
        
        Returns:
            Resulting hypervector
        """
        # Simple implementation - can be extended with full parser
        result = None
        current_op = None
        
        for token in expression.split():
            if token in concepts:
                vec = concepts[token]
                if result is None:
                    result = vec
                elif current_op == '*':
                    result = self.bind(result, vec)
                elif current_op == '+':
                    result = self.bundle([result, vec])
            elif token in ['*', '+']:
                current_op = token
        
        return result if result is not None else self.create_random_vector()


class FractalMemory:
    """
    Fractal Memory System - Recursive summarization with holographic properties.
    
    Implements the "Fractal Summarizer" concept where:
    - Day vectors compress into week vectors
    - Week vectors compress into month vectors
    - Month vectors compress into year vectors
    - Query decompression works recursively
    """
    
    def __init__(self, hd_dimension: int = 10000, save_path: Optional[str] = None):
        self.hd_dimension = hd_dimension
        self.hdv = HyperdimensionalVector(hd_dimension)
        self.save_path = Path(save_path) if save_path else Path("./fractal_memory")
        self.save_path.mkdir(exist_ok=True)
        
        # Hierarchical memory structure
        self.memory_tree = {
            'daily': {},      # Individual day vectors
            'weekly': {},     # Week summary vectors
            'monthly': {},    # Month summary vectors
            'yearly': {}      # Year summary vectors
        }
        
        # Concept library (for holographic queries)
        self.concepts = {}
        
    def encode_text_to_hypervector(self, text: str) -> torch.Tensor:
        """
        Encode text into a hyperdimensional vector.
        Simple approach: bind character/word vectors together.
        """
        # Create random seed based on hash
        seed = hash(text) % (2**32)
        torch.manual_seed(seed)
        
        # For simplicity, create a deterministic vector from text
        base_vector = self.hdv.create_random_vector()
        
        # Bind character information
        for i, char in enumerate(text[:100]):  # Limit to 100 chars
            char_vector = torch.sign(torch.randn(self.hd_dimension))
            base_vector = self.hdv.bind(base_vector, self.hdv.permute_vector(char_vector, i))
        
        return torch.sign(base_vector)
    
    def store_daily_memory(self, date: str, content: str, metadata: Optional[Dict] = None):
        """
        Store a daily memory (e.g., journal entry, generated music, lyrics).
        
        Args:
            date: Date string (e.g., "2025-01-15")
            content: Content to store (text, description)
            metadata: Additional metadata
        """
        # Encode content as hypervector
        hypervector = self.encode_text_to_hypervector(content)
        
        self.memory_tree['daily'][date] = {
            'vector': hypervector,
            'content': content,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Stored daily memory for {date}")
        
        # Auto-compress if we have enough days
        self._auto_compress()
    
    def _auto_compress(self):
        """Automatically compress memories into higher-level summaries"""
        # Compress daily -> weekly
        daily_dates = sorted(self.memory_tree['daily'].keys())
        
        if len(daily_dates) >= 7:
            # Group by weeks
            week_groups = {}
            for date in daily_dates:
                week_id = self._get_week_id(date)
                if week_id not in week_groups:
                    week_groups[week_id] = []
                week_groups[week_id].append(date)
            
            # Compress each complete week
            for week_id, dates in week_groups.items():
                if len(dates) >= 7 and week_id not in self.memory_tree['weekly']:
                    self._compress_to_weekly(week_id, dates)
    
    def _get_week_id(self, date_str: str) -> str:
        """Get week identifier from date (e.g., "2025-W03")"""
        from datetime import datetime
        date = datetime.strptime(date_str, "%Y-%m-%d")
        return f"{date.year}-W{date.isocalendar()[1]:02d}"
    
    def _compress_to_weekly(self, week_id: str, daily_dates: List[str]):
        """Compress daily memories into a weekly summary vector"""
        daily_vectors = [
            self.memory_tree['daily'][date]['vector']
            for date in daily_dates
            if date in self.memory_tree['daily']
        ]
        
        if not daily_vectors:
            return
        
        # Bundle all daily vectors into weekly vector
        weekly_vector = self.hdv.bundle(daily_vectors)
        
        self.memory_tree['weekly'][week_id] = {
            'vector': weekly_vector,
            'dates': daily_dates,
            'compressed_from': len(daily_vectors),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Compressed {len(daily_vectors)} days into week {week_id}")
    
    def query_memory(
        self,
        query: str,
        level: str = 'all',
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query the fractal memory system.
        
        Args:
            query: Query text
            level: Memory level to search ('daily', 'weekly', 'monthly', 'yearly', 'all')
            top_k: Number of top results to return
        
        Returns:
            List of matching memories with similarity scores
        """
        query_vector = self.encode_text_to_hypervector(query)
        
        results = []
        
        levels_to_search = [level] if level != 'all' else ['daily', 'weekly', 'monthly', 'yearly']
        
        for mem_level in levels_to_search:
            for key, memory in self.memory_tree[mem_level].items():
                similarity = self.hdv.similarity(query_vector, memory['vector'])
                results.append({
                    'level': mem_level,
                    'id': key,
                    'similarity': similarity,
                    'memory': memory
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
    
    def save_to_disk(self):
        """Save the fractal memory to disk"""
        save_file = self.save_path / "fractal_memory.pt"
        
        # Convert to serializable format
        save_data = {
            'hd_dimension': self.hd_dimension,
            'memory_tree': {},
            'concepts': {}
        }
        
        # Save memory tree
        for level, memories in self.memory_tree.items():
            save_data['memory_tree'][level] = {}
            for key, memory in memories.items():
                save_data['memory_tree'][level][key] = {
                    'vector': memory['vector'].cpu().numpy().tolist(),
                    'content': memory.get('content', ''),
                    'metadata': memory.get('metadata', {}),
                    'timestamp': memory.get('timestamp', ''),
                    'dates': memory.get('dates', []),
                    'compressed_from': memory.get('compressed_from', 0)
                }
        
        # Save using torch
        torch.save(save_data, save_file)
        logger.info(f"Saved fractal memory to {save_file}")
    
    def load_from_disk(self):
        """Load the fractal memory from disk"""
        save_file = self.save_path / "fractal_memory.pt"
        
        if not save_file.exists():
            logger.warning(f"No saved memory found at {save_file}")
            return
        
        save_data = torch.load(save_file)
        
        self.hd_dimension = save_data['hd_dimension']
        
        # Restore memory tree
        for level, memories in save_data['memory_tree'].items():
            self.memory_tree[level] = {}
            for key, memory in memories.items():
                self.memory_tree[level][key] = {
                    'vector': torch.tensor(memory['vector']),
                    'content': memory.get('content', ''),
                    'metadata': memory.get('metadata', {}),
                    'timestamp': memory.get('timestamp', ''),
                    'dates': memory.get('dates', []),
                    'compressed_from': memory.get('compressed_from', 0)
                }
        
        logger.info(f"Loaded fractal memory from {save_file}")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the memory system"""
        return {
            'daily_memories': len(self.memory_tree['daily']),
            'weekly_memories': len(self.memory_tree['weekly']),
            'monthly_memories': len(self.memory_tree['monthly']),
            'yearly_memories': len(self.memory_tree['yearly']),
            'total_concepts': len(self.concepts),
            'hd_dimension': self.hd_dimension
        }


# Example usage
if __name__ == "__main__":
    # Initialize fractal memory
    fractal_memory = FractalMemory(hd_dimension=10000)
    
    # Store some daily memories
    fractal_memory.store_daily_memory(
        "2025-01-15",
        "Generated a funky jazz tune with saxophone. Very upbeat and energetic.",
        metadata={"genre": "jazz", "mood": "upbeat"}
    )
    
    fractal_memory.store_daily_memory(
        "2025-01-16",
        "Created a melancholic piano ballad about lost love.",
        metadata={"genre": "ballad", "mood": "sad"}
    )
    
    # Query the memory
    results = fractal_memory.query_memory("jazz music", top_k=3)
    
    print("\nQuery Results:")
    for result in results:
        print(f"Level: {result['level']}, ID: {result['id']}, Similarity: {result['similarity']:.3f}")
    
    # Save to disk
    fractal_memory.save_to_disk()
    
    # Statistics
    print("\nMemory Statistics:")
    print(json.dumps(fractal_memory.get_statistics(), indent=2))
