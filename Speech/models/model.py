# Add this at the beginning of your script to disable Flash Attention
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force transformers to not use Flash Attention
import transformers
transformers.utils.is_flash_attn_available = lambda: False

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple

from .text_encoder import TextEncoder
from .speaker_encoder import SpeakerEncoder
from .decoder import MelSpectrogram2Decoder
from .vocoder import HiFiGANVocoder

# Import emotion encoder if it exists
try:
    from .emotion_encoder import EmotionEncoder
    has_emotion_encoder = True
except ImportError:
    has_emotion_encoder = False

class VoiceCloningModel(nn.Module):
    """
    Complete voice cloning model that combines text encoding, speaker encoding, 
    decoding and vocoding
    """
    def __init__(self, config: Dict = None):
        super().__init__()
        self.config = config or {}
        
        # Get parameters from config with defaults
        self.d_model = self.config.get('d_model', 512)
        self.use_emotion_encoder = self.config.get('use_emotion_encoder', False)
        
        # Initialize text encoder with appropriate parameters
        text_config = self.config.get('text_encoder', {})
        # Update TextEncoder initialization
        self.text_encoder = TextEncoder(
            output_dim=self.d_model,
            max_text_length=text_config.get('max_text_length', 200)
        )
        
        # Initialize speaker encoder
        self.speaker_encoder = SpeakerEncoder(self.config.get('speaker_encoder', {}))
        
        # Initialize emotion encoder if needed and available
        if self.use_emotion_encoder and has_emotion_encoder:
            emotion_config = self.config.get('emotion_encoder', {})
            self.emotion_encoder = EmotionEncoder(
                emotion_dim=emotion_config.get('emotion_dim', self.d_model)
            )
        
        # Initialize decoder
        decoder_config = self.config.get('decoder', {})
        self.decoder = MelSpectrogram2Decoder(
            d_model=self.d_model,
            nhead=decoder_config.get('nhead', 8),
            num_decoder_layers=decoder_config.get('num_decoder_layers', 6),
            dim_feedforward=decoder_config.get('dim_feedforward', 2048),
            dropout=decoder_config.get('dropout', 0.1),
            mel_channels=decoder_config.get('mel_channels', 80)
        )
        
        # Initialize vocoder
        self.vocoder = HiFiGANVocoder(self.config.get('vocoder', {}))
        
    def forward(self,
                text: List[str],
                speaker_embeddings: torch.Tensor,
                emotion_labels: Optional[List[str]] = None,
                mel_targets: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the voice cloning model
        
        Args:
            text: List of text strings to synthesize
            speaker_embeddings: Speaker embeddings (batch_size, speaker_dim)
            emotion_labels: Optional emotion labels
            mel_targets: Optional mel spectrogram targets for training
            teacher_forcing_ratio: Teacher forcing ratio for training
            
        Returns:
            mel_outputs: Generated mel spectrograms
            stop_outputs: Stop token predictions
            waveform: Generated audio waveforms (inference only)
        """
        # Encode text
        text_memory = self.text_encoder(text)
        
        # Process emotion if available
        emotion_embeddings = None
        if self.use_emotion_encoder and hasattr(self, 'emotion_encoder') and emotion_labels is not None:
            emotion_embeddings = self.emotion_encoder(emotion_labels)
        
        # Decode to mel spectrogram
        if mel_targets is not None:
            # Training mode
            mel_outputs, stop_outputs = self.decoder(
                text_memory=text_memory,
                speaker_embedding=speaker_embeddings,
                emotion_embedding=emotion_embeddings,
                mel_targets=mel_targets,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            return mel_outputs, stop_outputs, None
        else:
            # Inference mode
            mel_outputs, stop_outputs = self.decoder.inference(
                text_memory=text_memory,
                speaker_embedding=speaker_embeddings,
                emotion_embedding=emotion_embeddings
            )
            
            # Generate waveform with vocoder
            waveform = self.vocoder.inference(mel_outputs)
            
            return mel_outputs, stop_outputs, waveform


class VoiceCloningLoss(nn.Module):
    """Loss function for voice cloning model"""
    
    def __init__(self):
        super().__init__()
        self.mel_loss = nn.MSELoss()
        self.stop_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, 
                mel_outputs: torch.Tensor, 
                stop_outputs: torch.Tensor,
                mel_targets: torch.Tensor, 
                stop_targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the total loss
        
        Args:
            mel_outputs: Predicted mel spectrograms
            stop_outputs: Predicted stop tokens
            mel_targets: Target mel spectrograms
            stop_targets: Target stop tokens
            
        Returns:
            total_loss: Combined loss value
        """
        mel_loss = self.mel_loss(mel_outputs, mel_targets)
        stop_loss = self.stop_loss(stop_outputs, stop_targets)
        
        # Weighted sum of losses
        total_loss = mel_loss + stop_loss
        
        return total_loss