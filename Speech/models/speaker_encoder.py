"""
Speaker Encoder Module

Extracts speaker embeddings from reference audio using the Resemblyzer VoiceEncoder.
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from pathlib import Path
from typing import List, Union, Optional
from resemblyzer import VoiceEncoder, preprocess_wav

class SpeakerEncoder:
    """
    Speaker encoder that extracts voice embeddings from audio samples
    """
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the speaker encoder
        
        Args:
            device: Device to run the model on (default: CPU or GPU if available)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.encoder = VoiceEncoder().to(device)
        print(f"Speaker encoder initialized on {device}")
        
    def compute_embedding(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Extract speaker embedding from audio file
        
        Args:
            audio_path: Path to reference audio file
            
        Returns:
            Speaker embedding as numpy array (shape: [256])
        """
        # Check if file exists
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Process audio file using resemblyzer's preprocessing
        try:
            wav = preprocess_wav(str(audio_path))
            
            # Extract embedding
            embedding = self.encoder.embed_utterance(wav)
            return embedding
        except Exception as e:
            raise RuntimeError(f"Error processing audio file {audio_path}: {str(e)}")
    
    def compute_embeddings_batch(self, audio_paths: List[Union[str, Path]]) -> np.ndarray:
        """
        Extract speaker embeddings from multiple audio files
        
        Args:
            audio_paths: List of paths to reference audio files
            
        Returns:
            Array of speaker embeddings (shape: [n_files, 256])
        """
        embeddings = []
        for path in audio_paths:
            embedding = self.compute_embedding(path)
            embeddings.append(embedding)
        
        return np.stack(embeddings)
    
    def compute_mean_embedding(self, audio_paths: List[Union[str, Path]]) -> np.ndarray:
        """
        Compute average embedding from multiple audio files of the same speaker
        
        Args:
            audio_paths: List of paths to reference audio files
            
        Returns:
            Average speaker embedding (shape: [256])
        """
        embeddings = self.compute_embeddings_batch(audio_paths)
        return np.mean(embeddings, axis=0)