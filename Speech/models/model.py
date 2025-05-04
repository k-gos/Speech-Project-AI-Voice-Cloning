"""
Main voice cloning model integrating all components
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Import model components
from .text_encoder import TextEncoder
from .speaker_encoder import SpeakerEncoder
from .emotion_encoder import EmotionEncoder
from .decoder import MelSpectrogram2Decoder as MelSpecDecoder
from .vocoder import HiFiGANVocoder

class VoiceCloningModel(nn.Module):
    """
    Complete voice cloning model integrating all components
    """
    def __init__(self, 
                 d_model: int = 512,
                 speaker_embedding_dim: int = 256,
                 emotion_embedding_dim: int = 128,
                 mel_channels: int = 80,
                 use_pretrained_speaker_encoder: bool = True,
                 speaker_encoder_path: Optional[str] = None,
                 use_pretrained_vocoder: bool = True,
                 vocoder_path: Optional[str] = None):
        """
        Initialize voice cloning model
        
        Args:
            d_model: Model dimension
            speaker_embedding_dim: Speaker embedding dimension
            emotion_embedding_dim: Emotion embedding dimension
            mel_channels: Number of mel spectrogram channels
            use_pretrained_speaker_encoder: Whether to use pretrained speaker encoder
            speaker_encoder_path: Path to pretrained speaker encoder
            use_pretrained_vocoder: Whether to use pretrained vocoder
            vocoder_path: Path to pretrained vocoder
        """
        super().__init__()
        
        self.d_model = d_model
        self.speaker_embedding_dim = speaker_embedding_dim
        self.emotion_embedding_dim = emotion_embedding_dim
        self.mel_channels = mel_channels
        
        # Initialize text encoder
        self.text_encoder = TextEncoder(d_model=d_model, use_bert=True)
        
        # Initialize speaker encoder
        self.speaker_encoder = SpeakerEncoder(
            embedding_dim=speaker_embedding_dim,
            use_pretrained=use_pretrained_speaker_encoder,
            pretrained_path=speaker_encoder_path
        )
        
        # Initialize emotion encoder
        self.emotion_encoder = EmotionEncoder(embedding_dim=emotion_embedding_dim)
        
        # Initialize decoder
        self.decoder = MelSpecDecoder(
            d_model=d_model,
            speaker_embedding_dim=speaker_embedding_dim,
            emotion_embedding_dim=emotion_embedding_dim,
            mel_channels=mel_channels
        )
        
        # Initialize vocoder
        self.vocoder = HiFiGANVocoder(
            mel_channels=mel_channels,
            use_pretrained=use_pretrained_vocoder,
            pretrained_path=vocoder_path
        )
        
        # Freeze speaker encoder if using pretrained
        if use_pretrained_speaker_encoder:
            for param in self.speaker_encoder.parameters():
                param.requires_grad = False
                
        # Freeze vocoder if using pretrained
        if use_pretrained_vocoder:
            for param in self.vocoder.parameters():
                param.requires_grad = False
    
    def forward(self, 
                text: Union[str, List[str]],
                speaker_embedding: Optional[torch.Tensor] = None,
                emotion: str = "neutral",
                emotion_reference: Optional[Union[str, np.ndarray]] = None,
                target_mel: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            text: Input text or list of texts
            speaker_embedding: Speaker embedding (if None, extracted from target_mel)
            emotion: Target emotion
            emotion_reference: Reference audio for emotion (path or waveform)
            target_mel: Target mel spectrogram (for teacher forcing in training)
            
        Returns:
            Dictionary with model outputs
        """
        # Encode text
        text_memory, text_mask = self.text_encoder(text)
        
        # Get speaker embedding if not provided
        if speaker_embedding is None and target_mel is not None:
            # Extract from target mel
            speaker_embedding = self.speaker_encoder(target_mel)
        
        if speaker_embedding is None:
            raise ValueError("Speaker embedding or target_mel must be provided")
            
        # Get emotion embedding
        emotion_embedding = self.emotion_encoder.get_emotion_embedding(
            emotion=emotion,
            acoustic_reference=emotion_reference
        )
        
        # Generate mel spectrogram with decoder
        mel_output = self.decoder(
            text_memory=text_memory,
            text_mask=text_mask,
            speaker_embedding=speaker_embedding,
            emotion_embedding=emotion_embedding,
            target_mel=target_mel
        )
        
        # Generate waveform with vocoder if not training
        waveform = None
        if target_mel is None:  # Inference mode
            waveform = self.vocoder(mel_output)
        
        return {
            "text_memory": text_memory,
            "text_mask": text_mask,
            "speaker_embedding": speaker_embedding,
            "emotion_embedding": emotion_embedding,
            "mel_output": mel_output,
            "waveform": waveform
        }
    
    def clone_voice(self,
                   text: str,
                   reference_audio: Union[str, np.ndarray],
                   emotion: str = "neutral",
                   emotion_reference: Optional[Union[str, np.ndarray]] = None) -> np.ndarray:
        """
        Clone voice and generate speech
        
        Args:
            text: Text to synthesize
            reference_audio: Reference audio for speaker (path or waveform)
            emotion: Target emotion
            emotion_reference: Reference audio for emotion (path or waveform)
            
        Returns:
            Generated waveform
        """
        # Extract speaker embedding from reference audio
        speaker_embedding = self.speaker_encoder.extract_embedding(reference_audio)
        
        # Forward pass with speaker embedding
        outputs = self.forward(
            text=text,
            speaker_embedding=speaker_embedding,
            emotion=emotion,
            emotion_reference=emotion_reference
        )
        
        # Return waveform
        return outputs["waveform"].squeeze(0).cpu().numpy()


class VoiceCloningLoss(nn.Module):
    """
    Loss function for voice cloning model
    """
    def __init__(self, 
                 mel_loss_weight: float = 1.0,
                 feature_loss_weight: float = 0.1):
        """
        Initialize loss function
        
        Args:
            mel_loss_weight: Weight for mel spectrogram reconstruction loss
            feature_loss_weight: Weight for feature matching loss
        """
        super().__init__()
        self.mel_loss_weight = mel_loss_weight
        self.feature_loss_weight = feature_loss_weight
        
    def forward(self, 
                predicted_mel: torch.Tensor, 
                target_mel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate loss
        
        Args:
            predicted_mel: Predicted mel spectrogram [batch, time, mel_channels]
            target_mel: Target mel spectrogram [batch, time, mel_channels]
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # L1 loss for mel reconstruction
        mel_l1_loss = F.l1_loss(predicted_mel, target_mel)
        
        # Mean squared error for spectrogram
        mel_mse_loss = F.mse_loss(predicted_mel, target_mel)
        
        # Feature matching loss (simplified version)
        feature_loss = torch.zeros_like(mel_l1_loss)
        
        # Calculate total loss
        total_loss = (
            self.mel_loss_weight * (mel_l1_loss + mel_mse_loss) + 
            self.feature_loss_weight * feature_loss
        )
        
        return total_loss, {
            "mel_l1_loss": mel_l1_loss,
            "mel_mse_loss": mel_mse_loss,
            "feature_loss": feature_loss
        }