"""
Model Selector - Cognitive Architecture Model Registry
Provides a unified interface for selecting and managing different model architectures
treating them as cognitive components rather than static tools.
"""
import torch
from typing import Dict, Any, Optional, Union, Type
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enumeration of available model architectures"""
    SONGBLOOM_ORIGINAL = "songbloom_original"
    DIFFUSION_TRANSFORMER = "diffusion_transformer"
    MUSICLDM = "musicldm"
    MUSICGEN = "musicgen"
    

class CognitiveLevel(Enum):
    """Cognitive Architecture Levels"""
    LEVEL_1_FOUNDATION = "foundation"  # Standard RAG
    LEVEL_2_HOLOGRAPHIC = "holographic"  # Hyperdimensional Computing
    LEVEL_3_ACTIVE_INFERENCE = "active_inference"  # Predictive System
    LEVEL_4_NEUROMORPHIC = "neuromorphic"  # Hardware-level optimization


class ModelRegistry:
    """
    Central registry for all available model architectures.
    Implements cognitive architecture patterns for model selection and management.
    """
    
    _models: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(
        cls,
        model_type: str,
        model_class: Type,
        cognitive_level: CognitiveLevel,
        description: str,
        default_config: Optional[Dict[str, Any]] = None
    ):
        """Register a new model architecture"""
        cls._models[model_type] = {
            'class': model_class,
            'cognitive_level': cognitive_level,
            'description': description,
            'config': default_config or {}
        }
        logger.info(f"Registered model: {model_type} at {cognitive_level.value}")
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """Get information about a registered model"""
        if model_type not in cls._models:
            raise ValueError(f"Model type '{model_type}' not registered")
        return cls._models[model_type]
    
    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all registered models"""
        return cls._models.copy()
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Create a model instance from the registry"""
        if model_type not in cls._models:
            raise ValueError(f"Model type '{model_type}' not registered")
        
        model_info = cls._models[model_type]
        model_class = model_info['class']
        
        # Merge default config with provided config
        final_config = {**model_info['config'], **(config or {}), **kwargs}
        
        logger.info(f"Creating {model_type} model with config: {final_config}")
        
        try:
            model = model_class(**final_config)
            return model
        except Exception as e:
            logger.error(f"Error creating model {model_type}: {str(e)}")
            raise


class ModelSelector:
    """
    Intelligent model selector that chooses the best model for a given task.
    Implements cognitive architecture principles for dynamic model selection.
    """
    
    def __init__(self, default_model: str = ModelType.SONGBLOOM_ORIGINAL.value):
        self.default_model = default_model
        self.current_model = None
        self.model_type = None
        
    def select_model(
        self,
        task: str,
        requirements: Optional[Dict[str, Any]] = None,
        cognitive_level: Optional[CognitiveLevel] = None
    ) -> str:
        """
        Intelligently select the best model for a given task.
        
        Args:
            task: Description of the task (e.g., "music_generation", "style_transfer")
            requirements: Dictionary of requirements (e.g., {"max_memory": "8GB", "speed": "fast"})
            cognitive_level: Desired cognitive architecture level
            
        Returns:
            Selected model type as string
        """
        requirements = requirements or {}
        
        # Task-based selection
        if task == "music_generation":
            if cognitive_level == CognitiveLevel.LEVEL_2_HOLOGRAPHIC:
                return ModelType.DIFFUSION_TRANSFORMER.value
            else:
                return ModelType.SONGBLOOM_ORIGINAL.value
        
        elif task == "text_to_music":
            if cognitive_level == CognitiveLevel.LEVEL_2_HOLOGRAPHIC:
                return ModelType.DIFFUSION_TRANSFORMER.value
            return ModelType.MUSICLDM.value
        
        # Default fallback
        return self.default_model
    
    def load_model(
        self,
        model_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Load a model from the registry"""
        if model_type is None:
            model_type = self.default_model
        
        self.current_model = ModelRegistry.create_model(model_type, config, **kwargs)
        self.model_type = model_type
        
        return self.current_model
    
    def get_current_model(self):
        """Get the currently loaded model"""
        return self.current_model
    
    def get_model_type(self) -> Optional[str]:
        """Get the type of the currently loaded model"""
        return self.model_type


# Initialize model registrations
def initialize_model_registry():
    """
    Initialize the model registry with available models.
    Call this function explicitly to avoid circular imports and enable easier testing.
    """
    
    # Import models dynamically to avoid circular imports
    try:
        from .diffusion_transformer import MusicDiffusionTransformer
        
        ModelRegistry.register(
            model_type=ModelType.DIFFUSION_TRANSFORMER.value,
            model_class=MusicDiffusionTransformer,
            cognitive_level=CognitiveLevel.LEVEL_2_HOLOGRAPHIC,
            description="Advanced Diffusion Transformer with Cognitive Architecture (Level 2)",
            default_config={
                'dim': 768,
                'num_layers': 12,
                'num_heads': 12,
                'vocab_size': 256,
                'mel_channels': 80,
                'max_seq_len': 4096
            }
        )
        logger.info("Registered MusicDiffusionTransformer")
    except Exception as e:
        logger.warning(f"Could not register MusicDiffusionTransformer: {e}")
    
    # Register SongBloom original
    try:
        from .songbloom.songbloom_pl import SongBloom_Sampler
        
        ModelRegistry.register(
            model_type=ModelType.SONGBLOOM_ORIGINAL.value,
            model_class=SongBloom_Sampler,
            cognitive_level=CognitiveLevel.LEVEL_1_FOUNDATION,
            description="Original SongBloom model (Level 1 - Foundation)",
            default_config={}
        )
        logger.info("Registered SongBloom_Sampler")
    except Exception as e:
        logger.warning(f"Could not register SongBloom_Sampler: {e}")


# Auto-initialize on import for convenience
# To avoid issues with circular imports or testing, you can disable this by
# setting SONGBLOOM_SKIP_AUTO_INIT environment variable
import os
if not os.environ.get('SONGBLOOM_SKIP_AUTO_INIT'):
    try:
        initialize_model_registry()
    except Exception as e:
        logger.warning(f"Auto-initialization failed: {e}. Call initialize_model_registry() manually.")
