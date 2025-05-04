"""
Decoder Module

Transforms encoded text, speaker, and emotion representations into mel spectrograms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with self-attention and cross-attention
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention with memory (encoder output)
        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward network
        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class MelSpectrogram2Decoder(nn.Module):
    """
    Decoder that generates mel spectrograms from text, speaker, and emotion embeddings
    using a transformer-based architecture
    """
    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_seq_len: int = 1000,
                 mel_channels: int = 80,
                 speaker_embedding_dim: int = 256,
                 emotion_embedding_dim: int = 128):
        """
        Initialize the decoder
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_decoder_layers: Number of transformer decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            mel_channels: Number of mel spectrogram channels
            speaker_embedding_dim: Dimension of speaker embeddings
            emotion_embedding_dim: Dimension of emotion embeddings
        """
        super().__init__()
        
        self.d_model = d_model
        self.mel_channels = mel_channels
        
        # Positional encoding for decoder inputs
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, d_model)
        )
        
        # Pre-net for autoregressive decoding
        self.pre_net = nn.Sequential(
            nn.Linear(mel_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, d_model)
        )
        
        # Project speaker and emotion embeddings to model dimension
        self.speaker_projection = nn.Linear(speaker_embedding_dim, d_model)
        self.emotion_projection = nn.Linear(emotion_embedding_dim, d_model)
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection to mel spectrogram
        self.output_projection = nn.Linear(d_model, mel_channels)
        
        # Stop token prediction
        self.stop_projection = nn.Linear(d_model, 1)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters"""
        # Initialize positional encodings
        position = torch.arange(0, self.positional_encoding.size(1)).unsqueeze(0).unsqueeze(2)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) * -(torch.log(torch.tensor(10000.0)) / self.d_model)
        )
        
        pos_enc = torch.zeros_like(self.positional_encoding[0])
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        
        self.positional_encoding.data.copy_(pos_enc)
        
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation"""
        mask = torch.triu(
            torch.ones((seq_len, seq_len), device=device) * float('-inf'),
            diagonal=1
        )
        return mask
    
    def forward(self, 
                text_memory: torch.Tensor,
                speaker_embedding: torch.Tensor,
                emotion_embedding: Optional[torch.Tensor] = None,
                mel_targets: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mel spectrograms from encoded inputs
        
        Args:
            text_memory: Text encoder outputs [batch_size, text_seq_len, d_model]
            speaker_embedding: Speaker embeddings [batch_size, speaker_embedding_dim]
            emotion_embedding: Emotion embeddings [batch_size, emotion_embedding_dim]
            mel_targets: Target mel spectrograms for training [batch_size, mel_seq_len, mel_channels]
            teacher_forcing_ratio: Ratio for teacher forcing (1.0 = always use targets)
            
        Returns:
            mel_outputs: Generated mel spectrograms [batch_size, mel_seq_len, mel_channels]
            stop_tokens: Stop token predictions [batch_size, mel_seq_len]
        """
        batch_size = text_memory.size(0)
        device = text_memory.device
        
        # Project speaker and emotion embeddings
        speaker_features = self.speaker_projection(speaker_embedding).unsqueeze(1)
        
        # Add emotion features if provided
        if emotion_embedding is not None:
            emotion_features = self.emotion_projection(emotion_embedding).unsqueeze(1)
            memory = torch.cat([speaker_features, emotion_features, text_memory], dim=1)
        else:
            memory = torch.cat([speaker_features, text_memory], dim=1)
        
        # Get memory sequence length
        memory_seq_len = memory.size(1)
        
        # Determine target sequence length
        if mel_targets is not None:
            target_seq_len = mel_targets.size(1)
        else:
            target_seq_len = 500  # Default length for inference
        
        # Initialize decoder input with zero frame
        decoder_input = torch.zeros(
            (batch_size, 1, self.mel_channels),
            device=device
        )
        
        # Initialize outputs
        mel_outputs = torch.zeros(
            (batch_size, target_seq_len, self.mel_channels),
            device=device
        )
        stop_outputs = torch.zeros(
            (batch_size, target_seq_len),
            device=device
        )
        
        # Create causal mask for self-attention
        causal_mask = self._create_causal_mask(target_seq_len, device)
        
        # Autoregressive decoding
        for t in range(target_seq_len):
            # Decide whether to use teacher forcing
            if mel_targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                if t == 0:
                    current_input = decoder_input
                else:
                    current_input = mel_targets[:, t-1:t, :]
            else:
                if t == 0:
                    current_input = decoder_input
                else:
                    current_input = mel_outputs[:, t-1:t, :]
            
            # Process through pre-net
            pre_out = self.pre_net(current_input)
            
            # Add positional encoding
            pos_emb = self.positional_encoding[:, t:t+1, :]
            decoder_out = pre_out + pos_emb
            
            # Apply decoder layers
            for layer in self.layers:
                decoder_out = layer(
                    decoder_out.transpose(0, 1),
                    memory.transpose(0, 1),
                    tgt_mask=causal_mask[t:t+1, :t+1]
                ).transpose(0, 1)
            
            # Project to mel spectrogram and stop token
            mel_frame = self.output_projection(decoder_out.squeeze(1))
            stop_logit = self.stop_projection(decoder_out.squeeze(1))
            
            # Store outputs
            mel_outputs[:, t, :] = mel_frame
            stop_outputs[:, t] = stop_logit.squeeze(-1)
            
        return mel_outputs, stop_outputs
    
    def inference(self,
                  text_memory: torch.Tensor,
                  speaker_embedding: torch.Tensor,
                  emotion_embedding: Optional[torch.Tensor] = None,
                  max_length: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mel spectrograms during inference
        
        Args:
            text_memory: Text encoder outputs
            speaker_embedding: Speaker embedding
            emotion_embedding: Emotion embedding (optional)
            max_length: Maximum output sequence length
            
        Returns:
            Generated mel spectrogram and stop tokens
        """
        # Use deterministic generation for inference
        with torch.no_grad():
            batch_size = text_memory.size(0)
            device = text_memory.device
            
            # Project speaker and emotion embeddings
            speaker_features = self.speaker_projection(speaker_embedding).unsqueeze(1)
            
            # Add emotion features if provided
            if emotion_embedding is not None:
                emotion_features = self.emotion_projection(emotion_embedding).unsqueeze(1)
                memory = torch.cat([speaker_features, emotion_features, text_memory], dim=1)
            else:
                memory = torch.cat([speaker_features, text_memory], dim=1)
            
            # Initialize decoder input with zero frame
            decoder_input = torch.zeros((batch_size, 1, self.mel_channels), device=device)
            
            # Initialize outputs
            mel_outputs = []
            stop_outputs = []
            
            # Autoregressive generation
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            for t in range(max_length):
                if t == 0:
                    current_input = decoder_input
                else:
                    current_input = mel_frame.unsqueeze(1)
                
                # Process through pre-net
                pre_out = self.pre_net(current_input)
                
                # Add positional encoding
                pos_emb = self.positional_encoding[:, t:t+1, :]
                decoder_out = pre_out + pos_emb
                
                # Apply decoder layers
                for layer in self.layers:
                    decoder_out = layer(
                        decoder_out.transpose(0, 1),
                        memory.transpose(0, 1)
                    ).transpose(0, 1)
                
                # Project to mel spectrogram and stop token
                mel_frame = self.output_projection(decoder_out.squeeze(1))
                stop_logit = self.stop_projection(decoder_out.squeeze(1))
                
                # Store outputs
                mel_outputs.append(mel_frame)
                stop_outputs.append(stop_logit.squeeze(-1))
                
                # Check stop condition
                stop_pred = torch.sigmoid(stop_logit.squeeze(-1)) > 0.5
                finished = finished | stop_pred
                
                if finished.all():
                    break
            
            # Stack outputs
            mel_outputs = torch.stack(mel_outputs, dim=1)  # [batch_size, seq_len, mel_channels]
            stop_outputs = torch.stack(stop_outputs, dim=1)  # [batch_size, seq_len]
            
            return mel_outputs, stop_outputs