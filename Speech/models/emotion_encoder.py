"""
Emotion Encoder Module

Encodes emotion labels or descriptions into embeddings for emotional speech synthesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Union, Optional

class EmotionEncoder(nn.Module):
    """
    Emotion encoder that maps emotion labels or descriptions to embeddings
    """
    def __init__(self, 
                 emotion_dim: int = 128,
                 emotion_classes: Optional[List[str]] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the emotion encoder
        
        Args:
            emotion_dim: Dimension of emotion embeddings
            emotion_classes: List of supported emotion classes
            device: Device to run the model on
        """
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        if emotion_classes is None:
            emotion_classes = ["neutral", "happy", "sad", "angry", "surprised", "fear"]
            
        self.device = device
        self.emotion_dim = emotion_dim
        self.emotion_classes = emotion_classes
        self.num_emotions = len(emotion_classes)
        
        # Create emotion embedding table
        self.emotion_embeddings = nn.Embedding(self.num_emotions, emotion_dim)
        
        # Initialize with variability to help differentiate emotions
        nn.init.normal_(self.emotion_embeddings.weight, mean=0.0, std=0.1)
        
        # For continuous emotion values
        self.emotion_mapper = nn.Sequential(
            nn.Linear(2, 64),  # Valence-Arousal space (2D)
            nn.ReLU(),
            nn.Linear(64, emotion_dim)
        )
        
        # Move model to device
        self.to(device)
        print(f"Emotion encoder initialized with {len(emotion_classes)} emotions on {device}")
        
    def forward(self, emotions: Union[List[str], torch.Tensor]) -> torch.Tensor:
        """
        Encode emotions into embeddings
        
        Args:
            emotions: List of emotion labels or tensor of emotion indices
            
        Returns:
            Emotion embeddings (shape: [batch_size, emotion_dim])
        """
        if isinstance(emotions, list):
            # Convert emotion labels to indices
            emotion_indices = torch.tensor([
                self.emotion_classes.index(e) if e in self.emotion_classes else 0
                for e in emotions
            ], device=self.device)
        else:
            # Use provided indices
            emotion_indices = emotions.to(self.device)
        
        # Get emotion embeddings
        embeddings = self.emotion_embeddings(emotion_indices)
        
        return embeddings
    
    def encode_valence_arousal(self, valence_arousal: torch.Tensor) -> torch.Tensor:
        """
        Encode emotions from valence-arousal values for more nuanced control
        
        Args:
            valence_arousal: Batch of valence-arousal pairs [batch_size, 2]
                             Values should be normalized to [-1, 1] range
        
        Returns:
            Emotion embeddings (shape: [batch_size, emotion_dim])
        """
        # Process valence-arousal values through the emotion mapper
        return self.emotion_mapper(valence_arousal)
    
    def interpolate_emotions(self, emotion1: str, emotion2: str, weight: float) -> torch.Tensor:
        """
        Interpolate between two emotions with given weight
        
        Args:
            emotion1: First emotion label
            emotion2: Second emotion label
            weight: Interpolation weight (0.0 to 1.0) - how much of emotion2 to use
            
        Returns:
            Interpolated emotion embedding
        """
        # Convert emotions to indices
        idx1 = self.emotion_classes.index(emotion1) if emotion1 in self.emotion_classes else 0
        idx2 = self.emotion_classes.index(emotion2) if emotion2 in self.emotion_classes else 0
        
        # Get embeddings
        emb1 = self.emotion_embeddings(torch.tensor([idx1], device=self.device))
        emb2 = self.emotion_embeddings(torch.tensor([idx2], device=self.device))
        
        # Interpolate
        interpolated = (1 - weight) * emb1 + weight * emb2
        
        return interpolated