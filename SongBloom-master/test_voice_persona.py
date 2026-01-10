"""
Unit tests for SongBloom Voice Cloning System
Tests VoiceModelRegistry, VoiceQualityMetrics, VoicePersona, VoiceCloner, PersonaManager
"""
import unittest
import tempfile
import os
import json
import numpy as np
import torch
import torchaudio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add SongBloom directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice_persona import (
    VoiceModelRegistry,
    VoiceQualityMetrics,
    VoicePersona,
    VoiceCloner,
    PersonaManager,
    apply_quality_preset
)


class TestVoiceModelRegistry(unittest.TestCase):
    """Test VoiceModelRegistry functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.registry = VoiceModelRegistry()
    
    def test_singleton_pattern(self):
        """Test that VoiceModelRegistry is a singleton"""
        registry2 = VoiceModelRegistry()
        self.assertIs(self.registry, registry2)
    
    def test_list_available_models(self):
        """Test listing available models"""
        models = self.registry.list_available_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        self.assertIn('ecapa_voxceleb', models)
    
    def test_get_model_info(self):
        """Test getting model information"""
        info = self.registry.get_model_info('ecapa_voxceleb')
        self.assertIsNotNone(info)
        self.assertIn('source', info)
        self.assertIn('embedding_dim', info)
        self.assertEqual(info['type'], 'speaker_recognition')
    
    def test_get_unknown_model(self):
        """Test getting info for unknown model"""
        info = self.registry.get_model_info('nonexistent_model')
        self.assertIsNone(info)


class TestVoiceQualityMetrics(unittest.TestCase):
    """Test VoiceQualityMetrics functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_audio(self, duration: float = 5.0, sample_rate: int = 16000, 
                         amplitude: float = 0.5) -> str:
        """Create a test audio file"""
        num_samples = int(duration * sample_rate)
        # Generate sine wave
        t = torch.linspace(0, duration, num_samples)
        signal = amplitude * torch.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        signal = signal.unsqueeze(0)  # Add channel dimension
        
        audio_path = os.path.join(self.temp_dir, f'test_{duration}s.wav')
        torchaudio.save(audio_path, signal, sample_rate)
        return audio_path
    
    def test_calculate_snr(self):
        """Test SNR calculation"""
        # Create clean signal
        signal = 0.5 * torch.sin(2 * np.pi * 440 * torch.linspace(0, 1, 16000))
        snr = VoiceQualityMetrics.calculate_snr(signal)
        self.assertGreater(snr, 10.0)  # Clean signal should have high SNR
    
    def test_validate_audio_quality_valid(self):
        """Test audio validation with valid audio"""
        audio_path = self.create_test_audio(duration=5.0, amplitude=0.5)
        is_valid, message = VoiceQualityMetrics.validate_audio_quality(audio_path)
        self.assertTrue(is_valid)
        self.assertIn('valid', message.lower())
    
    def test_validate_audio_quality_too_short(self):
        """Test audio validation with too short audio"""
        audio_path = self.create_test_audio(duration=1.0)
        is_valid, message = VoiceQualityMetrics.validate_audio_quality(
            audio_path, min_duration=3.0
        )
        self.assertFalse(is_valid)
        self.assertIn('too short', message.lower())
    
    def test_validate_audio_quality_too_long(self):
        """Test audio validation with too long audio"""
        audio_path = self.create_test_audio(duration=70.0)
        is_valid, message = VoiceQualityMetrics.validate_audio_quality(
            audio_path, max_duration=60.0
        )
        self.assertFalse(is_valid)
        self.assertIn('too long', message.lower())
    
    def test_validate_audio_quality_too_quiet(self):
        """Test audio validation with too quiet audio"""
        audio_path = self.create_test_audio(duration=5.0, amplitude=0.005)
        is_valid, message = VoiceQualityMetrics.validate_audio_quality(audio_path)
        self.assertFalse(is_valid)
        self.assertIn('silent', message.lower())


class TestVoicePersona(unittest.TestCase):
    """Test VoicePersona functionality"""
    
    def test_persona_creation(self):
        """Test creating a voice persona"""
        persona = VoicePersona("TestArtist")
        self.assertEqual(persona.name, "TestArtist")
        self.assertIsNotNone(persona.persona_id)
        self.assertEqual(len(persona.persona_id), 16)
        self.assertEqual(persona.quality_preset, 'balanced')
    
    def test_persona_validation_valid(self):
        """Test persona validation with valid data"""
        persona = VoicePersona("ValidArtist")
        persona.voice_embedding = np.random.randn(192)
        is_valid, issues = persona.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
    
    def test_persona_validation_empty_name(self):
        """Test persona validation with empty name"""
        persona = VoicePersona("")
        is_valid, issues = persona.validate()
        self.assertFalse(is_valid)
        self.assertIn("Persona name is empty", issues)
    
    def test_persona_validation_invalid_quality_preset(self):
        """Test persona validation with invalid quality preset"""
        persona = VoicePersona("TestArtist")
        persona.quality_preset = "invalid_preset"
        is_valid, issues = persona.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid quality preset" in issue for issue in issues))
    
    def test_persona_to_dict(self):
        """Test converting persona to dictionary"""
        persona = VoicePersona("TestArtist")
        persona.voice_embedding = np.random.randn(192)
        data = persona.to_dict()
        
        self.assertIsInstance(data, dict)
        self.assertEqual(data['name'], "TestArtist")
        self.assertIn('persona_id', data)
        self.assertIn('voice_embedding', data)
        self.assertIsInstance(data['voice_embedding'], list)
    
    def test_persona_from_dict(self):
        """Test creating persona from dictionary"""
        original = VoicePersona("TestArtist")
        original.voice_embedding = np.random.randn(192)
        data = original.to_dict()
        
        restored = VoicePersona.from_dict(data)
        self.assertEqual(restored.name, original.name)
        self.assertEqual(restored.persona_id, original.persona_id)
        np.testing.assert_array_almost_equal(
            restored.voice_embedding, 
            original.voice_embedding
        )


class TestVoiceCloner(unittest.TestCase):
    """Test VoiceCloner functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        # Mock the encoder since we may not have speechbrain installed
        self.cloner = VoiceCloner()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_audio(self, duration: float = 5.0) -> str:
        """Create a test audio file"""
        sample_rate = 16000
        num_samples = int(duration * sample_rate)
        t = torch.linspace(0, duration, num_samples)
        signal = 0.5 * torch.sin(2 * np.pi * 440 * t)
        signal = signal.unsqueeze(0)
        
        audio_path = os.path.join(self.temp_dir, f'test_{duration}s.wav')
        torchaudio.save(audio_path, signal, sample_rate)
        return audio_path
    
    def test_compute_voice_similarity(self):
        """Test computing similarity between embeddings"""
        embedding1 = np.random.randn(192)
        embedding2 = embedding1 + 0.1 * np.random.randn(192)
        
        similarity = self.cloner.compute_voice_similarity(embedding1, embedding2)
        self.assertGreater(similarity, 0.5)  # Similar embeddings
        self.assertLessEqual(similarity, 1.0)
    
    @patch('voice_persona.VOICE_CLONE_AVAILABLE', True)
    def test_create_persona_from_samples(self):
        """Test creating persona from audio samples"""
        # Create test audio files
        audio_files = [
            self.create_test_audio(5.0),
            self.create_test_audio(7.0),
        ]
        
        # Mock the encoder
        with patch.object(self.cloner, 'encoder') as mock_encoder:
            mock_embedding = torch.randn(1, 192)
            mock_encoder.encode_batch.return_value = mock_embedding
            
            persona = self.cloner.create_voice_persona_from_samples(
                "TestArtist",
                audio_files,
                "Test description"
            )
            
            self.assertEqual(persona.name, "TestArtist")
            self.assertEqual(persona.metadata['description'], "Test description")


class TestPersonaManager(unittest.TestCase):
    """Test PersonaManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = PersonaManager(personas_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_persona(self):
        """Test saving and loading persona"""
        # Create persona
        persona = VoicePersona("TestArtist")
        persona.voice_embedding = np.random.randn(192)
        
        # Save
        success = self.manager.save_persona(persona)
        self.assertTrue(success)
        
        # Load
        loaded = self.manager.load_persona(persona.persona_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, persona.name)
        self.assertEqual(loaded.persona_id, persona.persona_id)
    
    def test_save_invalid_persona(self):
        """Test saving invalid persona"""
        persona = VoicePersona("")  # Invalid empty name
        success = self.manager.save_persona(persona)
        self.assertFalse(success)
    
    def test_load_nonexistent_persona(self):
        """Test loading nonexistent persona"""
        loaded = self.manager.load_persona("nonexistent_id")
        self.assertIsNone(loaded)
    
    def test_list_personas(self):
        """Test listing personas"""
        # Create and save multiple personas
        for i in range(3):
            persona = VoicePersona(f"Artist{i}")
            persona.voice_embedding = np.random.randn(192)
            self.manager.save_persona(persona)
        
        # List personas
        personas = self.manager.list_personas()
        self.assertEqual(len(personas), 3)
        self.assertIsInstance(personas[0], dict)
        self.assertIn('name', personas[0])
        self.assertIn('persona_id', personas[0])
    
    def test_delete_persona(self):
        """Test deleting persona"""
        # Create and save persona
        persona = VoicePersona("TestArtist")
        persona.voice_embedding = np.random.randn(192)
        self.manager.save_persona(persona)
        
        # Delete
        success = self.manager.delete_persona(persona.persona_id)
        self.assertTrue(success)
        
        # Verify deletion
        loaded = self.manager.load_persona(persona.persona_id)
        self.assertIsNone(loaded)
    
    def test_export_and_import_persona(self):
        """Test exporting and importing persona"""
        # Create persona
        persona = VoicePersona("TestArtist")
        persona.voice_embedding = np.random.randn(192)
        self.manager.save_persona(persona)
        
        # Export
        export_path = os.path.join(self.temp_dir, "export.json")
        success = self.manager.export_persona(persona.persona_id, export_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(export_path))
        
        # Delete original
        self.manager.delete_persona(persona.persona_id)
        
        # Import
        imported = self.manager.import_persona(export_path)
        self.assertIsNotNone(imported)
        self.assertEqual(imported.name, persona.name)
        self.assertEqual(imported.persona_id, persona.persona_id)


class TestQualityPresets(unittest.TestCase):
    """Test quality preset functionality"""
    
    def test_apply_ultra_preset(self):
        """Test applying ultra quality preset"""
        settings = {}
        updated = apply_quality_preset(settings, 'ultra')
        
        self.assertEqual(updated['steps'], 100)
        self.assertEqual(updated['cfg_coef'], 2.0)
        self.assertEqual(updated['top_k'], 300)
    
    def test_apply_fast_preset(self):
        """Test applying fast quality preset"""
        settings = {}
        updated = apply_quality_preset(settings, 'fast')
        
        self.assertEqual(updated['steps'], 30)
        self.assertEqual(updated['cfg_coef'], 1.3)
        self.assertEqual(updated['top_k'], 150)
    
    def test_apply_invalid_preset(self):
        """Test applying invalid preset"""
        settings = {'existing_key': 'value'}
        updated = apply_quality_preset(settings, 'invalid')
        
        # Should return original settings
        self.assertEqual(updated['existing_key'], 'value')
        self.assertNotIn('steps', updated)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
