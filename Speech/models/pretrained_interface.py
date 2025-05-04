"""
Interface for loading and using pretrained models in the voice cloning system
"""

import os
import torch
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import requests
import tarfile
import zipfile
from tqdm import tqdm
import hashlib

# Default URLs for pretrained models
PRETRAINED_MODELS = {
    "speaker_encoder": {
        "url": "https://github.com/resemble-ai/resemblyzer/raw/master/resemblyzer/pretrained.pt",
        "md5": "0cdba6f77e5fa2c40ebd0e3557944313",
        "description": "Speaker encoder model based on GE2E loss"
    },
    "hifigan_vocoder": {
        "url": "https://github.com/jik876/hifi-gan/releases/download/v1/g_02500000.pt",
        "md5": "1d25c1b1f064bd11f358d9c48e58da8f",
        "description": "HiFi-GAN universal vocoder trained on multiple datasets"
    }
}

def download_file(url: str, dest_path: str) -> str:
    """
    Download file from URL to destination path with progress bar
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        
    Returns:
        Path to downloaded file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))
    
    return dest_path

def get_md5(file_path: str) -> str:
    """
    Calculate MD5 hash of a file
    
    Args:
        file_path: Path to file
        
    Returns:
        MD5 hash as hex string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_pretrained_model(model_type: str, models_dir: Optional[str] = None) -> str:
    """
    Get path to pretrained model, downloading if needed
    
    Args:
        model_type: Type of model ("speaker_encoder" or "hifigan_vocoder")
        models_dir: Directory to store pretrained models
        
    Returns:
        Path to pretrained model
    """
    if model_type not in PRETRAINED_MODELS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Use default directory if not provided
    if models_dir is None:
        models_dir = Path.home() / ".voice_cloning" / "pretrained"
    else:
        models_dir = Path(models_dir)
    
    # Create directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Get model info
    model_info = PRETRAINED_MODELS[model_type]
    model_url = model_info["url"]
    model_md5 = model_info["md5"]
    
    # Determine filename from URL
    filename = os.path.basename(model_url)
    model_path = models_dir / filename
    
    # Download if file doesn't exist or MD5 doesn't match
    if not model_path.exists() or get_md5(str(model_path)) != model_md5:
        print(f"Downloading {model_type} model...")
        download_file(model_url, str(model_path))
        
        # Verify MD5
        if get_md5(str(model_path)) != model_md5:
            raise ValueError(f"Downloaded model has incorrect MD5 hash")
    
    return str(model_path)

def load_speaker_encoder(path: Optional[str] = None) -> torch.nn.Module:
    """
    Load pretrained speaker encoder
    
    Args:
        path: Path to pretrained model, downloads default if None
        
    Returns:
        Speaker encoder model
    """
    # Get model path
    if path is None:
        path = get_pretrained_model("speaker_encoder")
    
    # Use custom SpeakerEncoder if available, otherwise use base Resemblyzer
    try:
        from models.speaker_encoder import SpeakerEncoder
        model = SpeakerEncoder()
        model.load_state_dict(torch.load(path))
        return model
    except ImportError:
        # Fallback to direct resemblyzer import
        from resemblyzer import VoiceEncoder
        return VoiceEncoder()

def load_vocoder(path: Optional[str] = None) -> torch.nn.Module:
    """
    Load pretrained vocoder
    
    Args:
        path: Path to pretrained model, downloads default if None
        
    Returns:
        Vocoder model
    """
    # Get model path
    if path is None:
        path = get_pretrained_model("hifigan_vocoder")
    
    # Load vocoder model
    from models.vocoder import HiFiGANGenerator
    
    # Initialize model
    vocoder = HiFiGANGenerator()
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    
    # Handle different checkpoint formats
    if "generator" in checkpoint:
        state_dict = checkpoint["generator"]
    else:
        state_dict = checkpoint
    
    # Load state dict
    vocoder.load_state_dict(state_dict)
    
    return vocoder