"""
Vocoder Module

Converts mel spectrograms into audio waveforms.
This implements a simplified HiFi-GAN vocoder architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import librosa

class ResBlock(nn.Module):
    """Residual block for the generator"""
    def __init__(self, 
                 channels: int, 
                 kernel_size: int = 3, 
                 dilation: int = 1, 
                 leaky_relu_alpha: float = 0.1):
        super().__init__()
        
        self.leaky_relu_alpha = leaky_relu_alpha
        padding = (kernel_size * dilation - dilation) // 2
        
        self.layers = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding),
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            res = x
            x = F.leaky_relu(x, self.leaky_relu_alpha)
            x = layer(x)
            x = x + res
        return x


class MRF(nn.Module):
    """Multi-Receptive Field Fusion module"""
    def __init__(self, 
                 channels: int,
                 kernel_sizes: List[int] = [3, 7, 11],
                 dilations: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 leaky_relu_alpha: float = 0.1):
        super().__init__()
        
        assert len(kernel_sizes) == len(dilations)
        
        self.resblock_groups = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            blocks = nn.ModuleList([
                ResBlock(channels, k, dil, leaky_relu_alpha) 
                for dil in d
            ])
            self.resblock_groups.append(blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = 0
        for blocks in self.resblock_groups:
            res = x
            for block in blocks:
                res = block(res)
            result = result + res
            
        # Average the outputs
        result = result / len(self.resblock_groups)
        return result


class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN Generator (simplified version)
    Converts mel spectrograms to waveforms.
    """
    def __init__(self,
                 in_channels: int = 80,
                 upsample_rates: List[int] = [8, 8, 2, 2],
                 upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
                 upsample_initial_channel: int = 512,
                 resblock_kernel_sizes: List[int] = [3, 7, 11],
                 resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 leaky_relu_alpha: float = 0.1):
        """
        Initialize HiFi-GAN Generator
        
        Args:
            in_channels: Number of input channels (mel bands)
            upsample_rates: Upsampling rates for each layer
            upsample_kernel_sizes: Kernel sizes for upsampling layers
            upsample_initial_channel: Initial number of channels for upsampling
            resblock_kernel_sizes: Kernel sizes for residual blocks
            resblock_dilation_sizes: Dilation sizes for residual blocks
            leaky_relu_alpha: Negative slope for LeakyReLU
        """
        super().__init__()
        
        self.num_upsamples = len(upsample_rates)
        self.leaky_relu_alpha = leaky_relu_alpha
        
        # Initial conv to convert mel-spec to latent representation
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3)
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        
        in_channels = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # Calculate output channels - reduce by factor of 2 for each layer
            out_channels = in_channels // 2
            
            # Ensure minimum number of channels
            out_channels = max(out_channels, 32)
            
            # Upsampling with transposed convolution
            self.ups.append(nn.ConvTranspose1d(
                in_channels,
                out_channels,
                k,
                stride=u,
                padding=(k-u)//2
            ))
            
            # Multi-receptive field fusion
            self.mrfs.append(MRF(
                out_channels,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                leaky_relu_alpha
            ))
            
            in_channels = out_channels
        
        # Final conv to generate waveform
        self.conv_post = nn.Conv1d(in_channels, 1, 7, padding=3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate waveform from mel spectrogram
        
        Args:
            x: Mel spectrogram [batch_size, n_mel, time]
            
        Returns:
            Audio waveform [batch_size, 1, time*upsample_factor]
        """
        # Initial convolution
        x = self.conv_pre(x)
        
        # Apply upsampling blocks
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.leaky_relu_alpha)
            x = self.ups[i](x)
            x = self.mrfs[i](x)
        
        # Final convolution
        x = F.leaky_relu(x, self.leaky_relu_alpha)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x


class Vocoder:
    """
    Wrapper class for the vocoder model with utilities for waveform generation
    """
    def __init__(self, model_path: Optional[str] = None, device: Optional[torch.device] = None):
        """
        Initialize the vocoder
        
        Args:
            model_path: Path to pretrained model weights
            device: Device to run the model on
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.device = device
        
        # Initialize the vocoder model
        self.model = HiFiGANGenerator().to(device)
        
        # Load pretrained weights if provided
        if model_path is not None:
            self.load_checkpoint(model_path)
            
        print(f"Vocoder initialized on {device}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model weights from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])
            elif "generator" in checkpoint:
                self.model.load_state_dict(checkpoint["generator"])
            else:
                self.model.load_state_dict(checkpoint)
                
            print(f"Loaded vocoder checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            
    def generate_waveform(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Convert mel spectrogram to audio waveform
        
        Args:
            mel_spectrogram: Mel spectrogram [batch_size, time, n_mels]
            
        Returns:
            Audio waveform [batch_size, samples]
        """
        self.model.eval()
        
        with torch.no_grad():
            # Ensure mel is on the correct device
            mel = mel_spectrogram.to(self.device)
            
            # Transpose for vocoder input [batch_size, n_mels, time]
            mel = mel.transpose(1, 2)
            
            # Generate audio
            waveform = self.model(mel)
            
            # Remove channel dimension and detach
            waveform = waveform.squeeze(1).cpu()
            
        return waveform
    
    def save_wav(self, waveform: torch.Tensor, path: str, sample_rate: int = 22050):
        """
        Save waveform as WAV file
        
        Args:
            waveform: Audio waveform [batch_size, samples] or [samples]
            path: Output file path
            sample_rate: Sample rate in Hz
        """
        # Ensure waveform is on CPU and convert to numpy
        if torch.is_tensor(waveform):
            waveform = waveform.cpu().numpy()
            
        # Remove batch dimension if present
        if len(waveform.shape) == 2:
            waveform = waveform[0]
            
        # Save as WAV
        librosa.output.write_wav(path, waveform, sample_rate)
        print(f"Saved audio to {path}")