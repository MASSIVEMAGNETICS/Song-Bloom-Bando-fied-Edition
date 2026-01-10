"""
SongBloom Voice Cloning & Persona Management System
Next-Gen X3: Voice cloning, persona save/load, and advanced customization
Enterprise-Grade Features: Model registry, caching, quality metrics, robust error handling
"""
import os
import json
import torch
import torchaudio
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import hashlib
from datetime import datetime
import numpy as np
import logging
from functools import lru_cache
import warnings

# Configure logging for enterprise deployment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from speechbrain.pretrained import EncoderClassifier
    VOICE_CLONE_AVAILABLE = True
except ImportError:
    VOICE_CLONE_AVAILABLE = False
    logger.warning("⚠️ Voice cloning requires speechbrain. Install with: pip install speechbrain")


class VoiceModelRegistry:
    """
    Registry for managing multiple voice models with dynamic loading capability
    Supports on-device and server-based model loading
    """
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VoiceModelRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.model_configs = {
                'ecapa_voxceleb': {
                    'source': 'speechbrain/spkrec-ecapa-voxceleb',
                    'type': 'speaker_recognition',
                    'embedding_dim': 192,
                    'sample_rate': 16000
                },
                'xvector': {
                    'source': 'speechbrain/spkrec-xvect-voxceleb',
                    'type': 'speaker_recognition', 
                    'embedding_dim': 512,
                    'sample_rate': 16000
                },
                'wavlm_base': {
                    'source': 'microsoft/wavlm-base-plus',
                    'type': 'speech_representation',
                    'embedding_dim': 768,
                    'sample_rate': 16000
                }
            }
            logger.info(f"Initialized VoiceModelRegistry with {len(self.model_configs)} model configurations")
    
    def get_model(self, model_name: str = 'ecapa_voxceleb') -> Optional[Any]:
        """
        Get or load a voice model with caching
        
        Args:
            model_name: Name of the model configuration
            
        Returns:
            Loaded model or None
        """
        if model_name in self._models:
            logger.debug(f"Using cached model: {model_name}")
            return self._models[model_name]
        
        if model_name not in self.model_configs:
            logger.error(f"Unknown model: {model_name}. Available: {list(self.model_configs.keys())}")
            return None
        
        try:
            config = self.model_configs[model_name]
            logger.info(f"Loading voice model: {model_name} from {config['source']}")
            
            if config['type'] == 'speaker_recognition' and VOICE_CLONE_AVAILABLE:
                model = EncoderClassifier.from_hparams(
                    source=config['source'],
                    savedir=f"voice_clone_models/{model_name}"
                )
                self._models[model_name] = model
                logger.info(f"✓ Successfully loaded {model_name}")
                return model
            else:
                logger.warning(f"Model type {config['type']} not yet supported or dependencies missing")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}", exc_info=True)
            return None
    
    def list_available_models(self) -> List[str]:
        """List all available model configurations"""
        return list(self.model_configs.keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a model"""
        return self.model_configs.get(model_name)


class VoiceQualityMetrics:
    """
    Quality metrics and validation for voice embeddings
    """
    
    @staticmethod
    def calculate_snr(audio_signal: torch.Tensor) -> float:
        """
        Calculate Signal-to-Noise Ratio
        
        Args:
            audio_signal: Audio tensor
            
        Returns:
            SNR in dB
        """
        try:
            signal_power = torch.mean(audio_signal ** 2)
            # Simple noise estimation using high-frequency components
            noise_estimate = torch.std(torch.diff(audio_signal))
            noise_power = noise_estimate ** 2
            
            if noise_power == 0:
                return float('inf')
            
            snr = 10 * torch.log10(signal_power / noise_power)
            return float(snr)
        except Exception as e:
            logger.warning(f"Could not calculate SNR: {e}")
            return 0.0
    
    @staticmethod
    def validate_audio_quality(audio_path: str, min_duration: float = 3.0, 
                               max_duration: float = 60.0) -> Tuple[bool, str]:
        """
        Validate audio file quality for voice cloning
        
        Args:
            audio_path: Path to audio file
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            signal, sr = torchaudio.load(audio_path)
            duration = signal.shape[1] / sr
            
            # Check duration
            if duration < min_duration:
                return False, f"Audio too short: {duration:.1f}s (minimum: {min_duration}s)"
            if duration > max_duration:
                return False, f"Audio too long: {duration:.1f}s (maximum: {max_duration}s)"
            
            # Check SNR
            snr = VoiceQualityMetrics.calculate_snr(signal)
            if snr < 10.0:
                return False, f"Audio quality too low (SNR: {snr:.1f}dB, minimum: 10dB)"
            
            # Check if audio is not silent
            if torch.max(torch.abs(signal)) < 0.01:
                return False, "Audio appears to be silent or too quiet"
            
            return True, f"Audio valid: {duration:.1f}s, SNR: {snr:.1f}dB"
            
        except Exception as e:
            return False, f"Error validating audio: {e}"


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
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate persona data integrity
        
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        if not self.name or not self.name.strip():
            issues.append("Persona name is empty")
        
        if self.voice_embedding is not None:
            if not isinstance(self.voice_embedding, (np.ndarray, list)):
                issues.append("Invalid voice embedding type")
            elif len(np.array(self.voice_embedding).shape) != 1:
                issues.append("Voice embedding must be 1-dimensional")
        
        if not self.quality_preset in ['ultra', 'high', 'balanced', 'fast']:
            issues.append(f"Invalid quality preset: {self.quality_preset}")
        
        return len(issues) == 0, issues
    
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
    Enterprise features: Multiple model support, caching, quality validation
    """
    def __init__(self, model_name: str = "ecapa_voxceleb"):
        self.model_name = model_name
        self.encoder = None
        self.registry = VoiceModelRegistry()
        self._embedding_cache = {}
        
        if VOICE_CLONE_AVAILABLE:
            try:
                self.encoder = self.registry.get_model(model_name)
                if self.encoder:
                    logger.info(f"✓ Voice cloning model '{model_name}' loaded")
                else:
                    logger.warning(f"Could not load model '{model_name}'")
            except Exception as e:
                logger.error(f"⚠️ Could not initialize voice cloning: {e}", exc_info=True)
    
    @lru_cache(maxsize=128)
    def _get_cached_embedding(self, audio_path: str) -> Optional[tuple]:
        """
        Get cached voice embedding (returns tuple for hashability)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Embedding as tuple or None
        """
        # This is a wrapper for LRU cache which requires hashable inputs
        embedding = self.extract_voice_embedding(audio_path, use_cache=False)
        if embedding is not None:
            return tuple(embedding.tolist())
        return None
    
    def extract_voice_embedding(self, audio_path: str, 
                               use_cache: bool = True,
                               validate_quality: bool = True) -> Optional[np.ndarray]:
        """
        Extract voice embedding from audio file with quality validation
        
        Args:
            audio_path: Path to audio file
            use_cache: Whether to use cached embeddings
            validate_quality: Whether to validate audio quality
            
        Returns:
            Voice embedding vector
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None
        
        # Use cache if enabled
        if use_cache and audio_path in self._embedding_cache:
            logger.debug(f"Using cached embedding for {audio_path}")
            return self._embedding_cache[audio_path]
        
        if self.encoder is None:
            logger.warning("⚠️ Voice cloning not available - encoder not loaded")
            return None
        
        # Validate audio quality
        if validate_quality:
            is_valid, msg = VoiceQualityMetrics.validate_audio_quality(audio_path)
            if not is_valid:
                logger.warning(f"Audio quality issue: {msg}")
                # Continue anyway but log the warning
        
        try:
            # Load audio
            signal, sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                signal = resampler(signal)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.encoder.encode_batch(signal)
            
            result = embedding.squeeze().cpu().numpy()
            
            # Cache the result
            if use_cache:
                self._embedding_cache[audio_path] = result
            
            logger.info(f"✓ Extracted voice embedding from {os.path.basename(audio_path)}")
            return result
        
        except Exception as e:
            logger.error(f"⚠️ Error extracting voice embedding from {audio_path}: {e}", exc_info=True)
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
        description: str = "",
        validate_quality: bool = True
    ) -> VoicePersona:
        """
        Create a voice persona from multiple audio samples with validation
        
        Args:
            name: Persona name
            audio_samples: List of audio file paths
            description: Persona description
            validate_quality: Whether to validate audio quality
            
        Returns:
            Voice persona
        """
        logger.info(f"Creating voice persona '{name}' from {len(audio_samples)} samples")
        persona = VoicePersona(name)
        persona.metadata['description'] = description
        
        # Extract embeddings from all samples
        embeddings = []
        valid_samples = []
        
        for sample_path in audio_samples:
            try:
                embedding = self.extract_voice_embedding(
                    sample_path, 
                    validate_quality=validate_quality
                )
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_samples.append(sample_path)
                    persona.voice_samples.append(sample_path)
            except Exception as e:
                logger.warning(f"Failed to process sample {sample_path}: {e}")
        
        if embeddings:
            # Average embeddings for robustness
            persona.voice_embedding = np.mean(embeddings, axis=0)
            logger.info(f"✓ Created voice persona '{name}' from {len(embeddings)} valid samples")
            
            # Store quality metrics
            persona.metadata['embedding_samples_count'] = len(embeddings)
            persona.metadata['embedding_dim'] = len(persona.voice_embedding)
        else:
            logger.warning(f"⚠️ Could not extract any valid voice embeddings for persona '{name}'")
        
        return persona


class PersonaManager:
    """
    Manage voice personas - save, load, organize
    Enterprise features: Atomic operations, backup, validation
    """
    def __init__(self, personas_dir: str = "./personas"):
        self.personas_dir = Path(personas_dir)
        self.personas_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.personas_dir / "configs").mkdir(exist_ok=True)
        (self.personas_dir / "voice_samples").mkdir(exist_ok=True)
        (self.personas_dir / "style_prompts").mkdir(exist_ok=True)
        (self.personas_dir / "backups").mkdir(exist_ok=True)
        
        logger.info(f"PersonaManager initialized with directory: {self.personas_dir}")
    
    def save_persona(self, persona: VoicePersona, create_backup: bool = True) -> bool:
        """
        Save persona to disk with validation and optional backup
        
        Args:
            persona: Voice persona to save
            create_backup: Whether to create backup of existing persona
            
        Returns:
            Success status
        """
        # Validate persona before saving
        is_valid, issues = persona.validate()
        if not is_valid:
            logger.error(f"Cannot save invalid persona: {', '.join(issues)}")
            return False
        
        try:
            config_path = self.personas_dir / "configs" / f"{persona.persona_id}.json"
            
            # Create backup if file exists
            if create_backup and config_path.exists():
                backup_path = self.personas_dir / "backups" / f"{persona.persona_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                import shutil
                shutil.copy2(config_path, backup_path)
                logger.debug(f"Created backup at {backup_path}")
            
            # Write to temporary file first (atomic operation)
            temp_path = config_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(persona.to_dict(), f, indent=2)
            
            # Atomic rename
            temp_path.replace(config_path)
            
            logger.info(f"✓ Saved persona '{persona.name}' (ID: {persona.persona_id})")
            return True
        
        except Exception as e:
            logger.error(f"⚠️ Error saving persona: {e}", exc_info=True)
            return False
    
    def load_persona(self, persona_id: str) -> Optional[VoicePersona]:
        """
        Load persona from disk with validation
        
        Args:
            persona_id: Persona ID
            
        Returns:
            Voice persona or None
        """
        try:
            config_path = self.personas_dir / "configs" / f"{persona_id}.json"
            
            if not config_path.exists():
                logger.warning(f"⚠️ Persona {persona_id} not found")
                return None
            
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            persona = VoicePersona.from_dict(data)
            
            # Validate loaded persona
            is_valid, issues = persona.validate()
            if not is_valid:
                logger.warning(f"Loaded persona has validation issues: {', '.join(issues)}")
            
            logger.info(f"✓ Loaded persona '{persona.name}' (ID: {persona_id})")
            return persona
        
        except json.JSONDecodeError as e:
            logger.error(f"⚠️ Invalid JSON in persona file {persona_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"⚠️ Error loading persona: {e}", exc_info=True)
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
                logger.warning(f"⚠️ Error reading {config_file}: {e}")
        
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
                logger.info(f"✓ Deleted persona {persona_id}")
                return True
            else:
                logger.warning(f"⚠️ Persona {persona_id} not found")
                return False
        
        except Exception as e:
            logger.error(f"⚠️ Error deleting persona: {e}", exc_info=True)
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
            
            logger.info(f"✓ Exported persona to {export_path}")
            return True
        
        except Exception as e:
            logger.error(f"⚠️ Error exporting persona: {e}", exc_info=True)
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
            
            logger.info(f"✓ Imported persona '{persona.name}'")
            return persona
        
        except Exception as e:
            logger.error(f"⚠️ Error importing persona: {e}", exc_info=True)
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
