"""
SongBloom Voice Cloning & Persona Management System
Next-Gen X3: Voice cloning, persona save/load, and advanced customization
"""
import os
import json
import torch
import torchaudio
from pathlib import Path
from typing import Optional, Dict, List, Any
import hashlib
from datetime import datetime
import numpy as np

try:
    from speechbrain.pretrained import EncoderClassifier
    VOICE_CLONE_AVAILABLE = True
except ImportError:
    VOICE_CLONE_AVAILABLE = False
    print("⚠️ Voice cloning requires speechbrain. Install with: pip install speechbrain")


class VoicePersona:
    """
    Voice persona with cloned voice characteristics and customization settings
    """
    def __init__(self, name: str, persona_id: Optional[str] = None):
        self.name = name
        self.persona_id = persona_id or self._generate_id()
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        
        # Voice characteristics
        self.voice_embedding = None
        self.voice_samples = []
        
        # Generation settings
        self.preferred_settings = {
            'cfg_coef': 1.5,
            'steps': 50,
            'top_k': 200,
            'temperature': 1.0,
            'style_strength': 1.0
        }
        
        # Style preferences
        self.style_tags = []
        self.genre_preferences = []
        
        # Quality settings
        self.quality_preset = 'balanced'  # ultra, high, balanced, fast
        
        # Metadata
        self.metadata = {
            'description': '',
            'use_count': 0,
            'total_generation_time': 0.0,
            'average_quality_score': 0.0,
            'tags': []
        }
    
    def _generate_id(self) -> str:
        """Generate unique persona ID"""
        timestamp = str(datetime.now().timestamp())
        hash_input = f"{self.name}_{timestamp}".encode()
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary"""
        return {
            'name': self.name,
            'persona_id': self.persona_id,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'voice_embedding': self.voice_embedding.tolist() if self.voice_embedding is not None else None,
            'voice_samples': self.voice_samples,
            'preferred_settings': self.preferred_settings,
            'style_tags': self.style_tags,
            'genre_preferences': self.genre_preferences,
            'quality_preset': self.quality_preset,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoicePersona':
        """Create persona from dictionary"""
        persona = cls(data['name'], data['persona_id'])
        persona.created_at = data['created_at']
        persona.updated_at = data['updated_at']
        
        if data['voice_embedding'] is not None:
            persona.voice_embedding = np.array(data['voice_embedding'])
        
        persona.voice_samples = data['voice_samples']
        persona.preferred_settings = data['preferred_settings']
        persona.style_tags = data['style_tags']
        persona.genre_preferences = data['genre_preferences']
        persona.quality_preset = data['quality_preset']
        persona.metadata = data['metadata']
        
        return persona


class VoiceCloner:
    """
    Advanced voice cloning system for creating personalized voice models
    """
    def __init__(self, model_name: str = "speechbrain/spkrec-ecapa-voxceleb"):
        self.model_name = model_name
        self.encoder = None
        
        if VOICE_CLONE_AVAILABLE:
            try:
                self.encoder = EncoderClassifier.from_hparams(
                    source=model_name,
                    savedir="voice_clone_models"
                )
                print("✓ Voice cloning model loaded")
            except Exception as e:
                print(f"⚠️ Could not load voice cloning model: {e}")
    
    def extract_voice_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extract voice embedding from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Voice embedding vector
        """
        if self.encoder is None:
            print("⚠️ Voice cloning not available")
            return None
        
        try:
            # Load audio
            signal, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                signal = resampler(signal)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.encoder.encode_batch(signal)
            
            return embedding.squeeze().cpu().numpy()
        
        except Exception as e:
            print(f"⚠️ Error extracting voice embedding: {e}")
            return None
    
    def compute_voice_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute similarity between two voice embeddings
        
        Args:
            embedding1: First voice embedding
            embedding2: Second voice embedding
            
        Returns:
            Similarity score (0-1)
        """
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return float(similarity)
    
    def create_voice_persona_from_samples(
        self,
        name: str,
        audio_samples: List[str],
        description: str = ""
    ) -> VoicePersona:
        """
        Create a voice persona from multiple audio samples
        
        Args:
            name: Persona name
            audio_samples: List of audio file paths
            description: Persona description
            
        Returns:
            Voice persona
        """
        persona = VoicePersona(name)
        persona.metadata['description'] = description
        
        # Extract embeddings from all samples
        embeddings = []
        for sample_path in audio_samples:
            embedding = self.extract_voice_embedding(sample_path)
            if embedding is not None:
                embeddings.append(embedding)
                persona.voice_samples.append(sample_path)
        
        if embeddings:
            # Average embeddings
            persona.voice_embedding = np.mean(embeddings, axis=0)
            print(f"✓ Created voice persona '{name}' from {len(embeddings)} samples")
        else:
            print(f"⚠️ Could not extract voice embeddings")
        
        return persona


class PersonaManager:
    """
    Manage voice personas - save, load, organize
    """
    def __init__(self, personas_dir: str = "./personas"):
        self.personas_dir = Path(personas_dir)
        self.personas_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.personas_dir / "configs").mkdir(exist_ok=True)
        (self.personas_dir / "voice_samples").mkdir(exist_ok=True)
        (self.personas_dir / "style_prompts").mkdir(exist_ok=True)
    
    def save_persona(self, persona: VoicePersona) -> bool:
        """
        Save persona to disk
        
        Args:
            persona: Voice persona to save
            
        Returns:
            Success status
        """
        try:
            config_path = self.personas_dir / "configs" / f"{persona.persona_id}.json"
            
            with open(config_path, 'w') as f:
                json.dump(persona.to_dict(), f, indent=2)
            
            print(f"✓ Saved persona '{persona.name}' to {config_path}")
            return True
        
        except Exception as e:
            print(f"⚠️ Error saving persona: {e}")
            return False
    
    def load_persona(self, persona_id: str) -> Optional[VoicePersona]:
        """
        Load persona from disk
        
        Args:
            persona_id: Persona ID
            
        Returns:
            Voice persona or None
        """
        try:
            config_path = self.personas_dir / "configs" / f"{persona_id}.json"
            
            if not config_path.exists():
                print(f"⚠️ Persona {persona_id} not found")
                return None
            
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            persona = VoicePersona.from_dict(data)
            print(f"✓ Loaded persona '{persona.name}'")
            return persona
        
        except Exception as e:
            print(f"⚠️ Error loading persona: {e}")
            return None
    
    def list_personas(self) -> List[Dict[str, Any]]:
        """
        List all available personas
        
        Returns:
            List of persona metadata
        """
        personas = []
        config_dir = self.personas_dir / "configs"
        
        for config_file in config_dir.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                
                personas.append({
                    'name': data['name'],
                    'persona_id': data['persona_id'],
                    'created_at': data['created_at'],
                    'use_count': data['metadata']['use_count'],
                    'quality_preset': data['quality_preset']
                })
            except Exception as e:
                print(f"⚠️ Error reading {config_file}: {e}")
        
        return sorted(personas, key=lambda x: x['created_at'], reverse=True)
    
    def delete_persona(self, persona_id: str) -> bool:
        """
        Delete persona
        
        Args:
            persona_id: Persona ID
            
        Returns:
            Success status
        """
        try:
            config_path = self.personas_dir / "configs" / f"{persona_id}.json"
            
            if config_path.exists():
                config_path.unlink()
                print(f"✓ Deleted persona {persona_id}")
                return True
            else:
                print(f"⚠️ Persona {persona_id} not found")
                return False
        
        except Exception as e:
            print(f"⚠️ Error deleting persona: {e}")
            return False
    
    def export_persona(self, persona_id: str, export_path: str) -> bool:
        """
        Export persona to a file
        
        Args:
            persona_id: Persona ID
            export_path: Export file path
            
        Returns:
            Success status
        """
        persona = self.load_persona(persona_id)
        if persona is None:
            return False
        
        try:
            with open(export_path, 'w') as f:
                json.dump(persona.to_dict(), f, indent=2)
            
            print(f"✓ Exported persona to {export_path}")
            return True
        
        except Exception as e:
            print(f"⚠️ Error exporting persona: {e}")
            return False
    
    def import_persona(self, import_path: str) -> Optional[VoicePersona]:
        """
        Import persona from a file
        
        Args:
            import_path: Import file path
            
        Returns:
            Voice persona or None
        """
        try:
            with open(import_path, 'r') as f:
                data = json.load(f)
            
            persona = VoicePersona.from_dict(data)
            self.save_persona(persona)
            
            print(f"✓ Imported persona '{persona.name}'")
            return persona
        
        except Exception as e:
            print(f"⚠️ Error importing persona: {e}")
            return None


def apply_quality_preset(settings: Dict[str, Any], preset: str) -> Dict[str, Any]:
    """
    Apply quality preset to generation settings
    
    Args:
        settings: Current settings
        preset: Quality preset (ultra, high, balanced, fast)
        
    Returns:
        Updated settings
    """
    presets = {
        'ultra': {
            'steps': 100,
            'cfg_coef': 2.0,
            'top_k': 300,
            'use_sampling': True,
            'optimization_level': 'standard'
        },
        'high': {
            'steps': 75,
            'cfg_coef': 1.8,
            'top_k': 250,
            'use_sampling': True,
            'optimization_level': 'standard'
        },
        'balanced': {
            'steps': 50,
            'cfg_coef': 1.5,
            'top_k': 200,
            'use_sampling': True,
            'optimization_level': 'standard'
        },
        'fast': {
            'steps': 30,
            'cfg_coef': 1.3,
            'top_k': 150,
            'use_sampling': True,
            'optimization_level': 'aggressive'
        }
    }
    
    if preset in presets:
        settings.update(presets[preset])
    
    return settings


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Persona Management")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create persona
    create_parser = subparsers.add_parser('create', help='Create voice persona')
    create_parser.add_argument('--name', type=str, required=True)
    create_parser.add_argument('--samples', type=str, nargs='+', required=True)
    create_parser.add_argument('--description', type=str, default='')
    
    # List personas
    list_parser = subparsers.add_parser('list', help='List personas')
    
    # Load persona
    load_parser = subparsers.add_parser('load', help='Load persona')
    load_parser.add_argument('--id', type=str, required=True)
    
    # Delete persona
    delete_parser = subparsers.add_parser('delete', help='Delete persona')
    delete_parser.add_argument('--id', type=str, required=True)
    
    # Export persona
    export_parser = subparsers.add_parser('export', help='Export persona')
    export_parser.add_argument('--id', type=str, required=True)
    export_parser.add_argument('--output', type=str, required=True)
    
    # Import persona
    import_parser = subparsers.add_parser('import', help='Import persona')
    import_parser.add_argument('--file', type=str, required=True)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        exit(0)
    
    manager = PersonaManager()
    cloner = VoiceCloner()
    
    if args.command == 'create':
        persona = cloner.create_voice_persona_from_samples(
            args.name,
            args.samples,
            args.description
        )
        manager.save_persona(persona)
        print(f"\n✓ Persona ID: {persona.persona_id}")
    
    elif args.command == 'list':
        personas = manager.list_personas()
        print(f"\nFound {len(personas)} personas:")
        for p in personas:
            print(f"  - {p['name']} (ID: {p['persona_id']}, used {p['use_count']} times)")
    
    elif args.command == 'load':
        persona = manager.load_persona(args.id)
        if persona:
            print(f"\nPersona: {persona.name}")
            print(f"  Quality: {persona.quality_preset}")
            print(f"  Settings: {persona.preferred_settings}")
    
    elif args.command == 'delete':
        manager.delete_persona(args.id)
    
    elif args.command == 'export':
        manager.export_persona(args.id, args.output)
    
    elif args.command == 'import':
        manager.import_persona(args.file)
