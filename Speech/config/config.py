"""
Configuration handling module for Voice Cloning project
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import argparse

class Config:
    """Configuration class for Voice Cloning system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration file (.yaml or .json)
        """
        # Default configuration
        self.config = {
            "audio": {
                "sampling_rate": 22050,
                "n_fft": 1024,
                "hop_length": 256,
                "win_length": 1024,
                "mel_channels": 80,
                "fmin": 0,
                "fmax": 8000,
                "max_audio_length": 10
            },
            "model": {
                "d_model": 512,
                "speaker_embedding_dim": 256,
                "emotion_embedding_dim": 128,
                "text_encoder_layers": 6,
                "text_encoder_heads": 8,
                "decoder_layers": 6,
                "decoder_heads": 8,
                "decoder_hidden_size": 1024,
                "vocoder_upsample_rates": [8, 8, 4, 2],
                "vocoder_upsample_kernel_sizes": [16, 16, 8, 4],
                "vocoder_upsample_initial_channel": 512,
                "vocoder_resblock_kernel_sizes": [3, 7, 11]
            },
            "training": {
                "batch_size": 16,
                "learning_rate": 0.0001,
                "weight_decay": 0.0001,
                "grad_clip_thresh": 1.0,
                "mel_loss_weight": 1.0,
                "feature_loss_weight": 0.1,
                "optimizer": "adam",
                "scheduler_patience": 3,
                "scheduler_factor": 0.5,
                "num_epochs": 100,
                "log_interval": 10,
                "save_interval": 5
            },
            "emotions": [
                "neutral",
                "happy",
                "sad",
                "angry",
                "surprised",
                "fear"
            ]
        }
        
        # Load configuration from file if provided
        if config_path is not None:
            self.load_config(config_path)
        
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        # Load based on file extension
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
        # Update configuration
        self._update_nested_dict(self.config, loaded_config)
        
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """
        Update nested dictionary recursively
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
        
    def save_config(self, config_path: str) -> None:
        """
        Save configuration to file
        
        Args:
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file extension
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value
        
        Args:
            key: Configuration key (dot notation for nested keys)
            value: Configuration value
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the last level
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        # Set the value
        config[keys[-1]] = value
        
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary syntax"""
        return self.get(key)
        
    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value using dictionary syntax"""
        self.set(key, value)
        
    def get_audio_params(self) -> Dict:
        """Get audio processing parameters"""
        return self.config.get("audio", {})
        
    def get_model_params(self) -> Dict:
        """Get model architecture parameters"""
        return self.config.get("model", {})
        
    def get_training_params(self) -> Dict:
        """Get training parameters"""
        return self.config.get("training", {})
        
    def get_supported_emotions(self) -> list:
        """Get list of supported emotions"""
        return self.config.get("emotions", [])

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training script"""
    parser = argparse.ArgumentParser(description="Voice Cloning System")
    
    # Required arguments
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config file (.yaml or .json)")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints and logs")
                       
    # Optional arguments
    parser.add_argument("--dataset", type=str, default="vctk",
                       choices=["libri_tts", "vctk", "common_voice", "aishell3", "ljspeech"],
                       help="Dataset name")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--speaker_encoder", type=str, default=None,
                       help="Path to pretrained speaker encoder")
    parser.add_argument("--vocoder", type=str, default=None,
                       help="Path to pretrained vocoder")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs to train (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--log_interval", type=int, default=None,
                       help="Logging interval in steps (overrides config)")
    parser.add_argument("--save_interval", type=int, default=None,
                       help="Checkpoint saving interval in epochs (overrides config)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--cpu", action="store_true",
                       help="Use CPU instead of GPU")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (fewer samples, more logging)")
    
    return parser.parse_args()