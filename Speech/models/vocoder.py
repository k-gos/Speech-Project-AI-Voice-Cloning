import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.convs = nn.ModuleList()
        
        for d in dilation:
            self.convs.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, padding=get_padding(kernel_size, d), dilation=d),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, padding=get_padding(kernel_size, 1))
            ))
            
    def forward(self, x):
        for conv in self.convs:
            x = x + conv(x)
        return x


class MRF(nn.Module):
    def __init__(self, channels, kernel_sizes=(3, 7, 11), dilations=((1, 3, 5), (1, 3, 5), (1, 3, 5))):
        super(MRF, self).__init__()
        self.resblocks = nn.ModuleList()
        
        for k, d in zip(kernel_sizes, dilations):
            self.resblocks.append(ResBlock(channels, k, d))
            
    def forward(self, x):
        output = 0
        for resblock in self.resblocks:
            output += resblock(x)
        return output / len(self.resblocks)


class HiFiGANVocoder(nn.Module):
    """HiFi-GAN vocoder for converting mel spectrograms to waveforms"""
    
    def __init__(self, config=None):
        super(HiFiGANVocoder, self).__init__()
        self.config = config or {}
        
        # Parameters
        self.in_channels = config.get("mel_channels", 80) if config else 80
        self.upsample_rates = config.get("upsample_rates", [8, 8, 2, 2]) if config else [8, 8, 2, 2]
        self.upsample_kernel_sizes = config.get("upsample_kernel_sizes", [16, 16, 4, 4]) if config else [16, 16, 4, 4]
        self.upsample_initial_channel = config.get("upsample_initial_channel", 512) if config else 512
        self.resblock_kernel_sizes = config.get("resblock_kernel_sizes", [3, 7, 11]) if config else [3, 7, 11]
        self.resblock_dilation_sizes = config.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]) if config else [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        
        # Initial conv
        self.conv_pre = nn.Conv1d(self.in_channels, self.upsample_initial_channel, kernel_size=7, padding=3)
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        
        in_channels = self.upsample_initial_channel
        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            self.ups.append(nn.ConvTranspose1d(
                in_channels, in_channels // 2, k, u, padding=(k-u)//2
            ))
            in_channels //= 2
            self.mrfs.append(MRF(in_channels, self.resblock_kernel_sizes, self.resblock_dilation_sizes))
            
        # Final conv
        self.conv_post = nn.Conv1d(in_channels, 1, kernel_size=7, padding=3)
        
        # Weight initialization
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            
    def forward(self, mel):
        """
        Args:
            mel: [B, n_mels, T]
            
        Returns:
            waveform: [B, 1, T*upsampling_factor]
        """
        x = self.conv_pre(mel)
        
        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = mrf(x)
            
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
        
    def inference(self, mel):
        """Generate waveform from mel spectrogram"""
        with torch.no_grad():
            return self.forward(mel)
    
    @classmethod
    def load_pretrained(cls, checkpoint_path=None):
        """Load pretrained weights"""
        model = cls()
        
        # If checkpoint_path is not provided, use default location
        if checkpoint_path is None:
            script_dir = Path(__file__).parent
            checkpoint_path = script_dir / "pretrained" / "vocoder" / "g_02500000.pt"
        
        # Check if file exists
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Vocoder checkpoint not found at {checkpoint_path}. Please run the download_models.py script first.")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'generator' in checkpoint:
            model.load_state_dict(checkpoint['generator'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model


def load_hifigan_vocoder(checkpoint_path=None):
    """Helper function to load pretrained HiFiGAN vocoder"""
    return HiFiGANVocoder.load_pretrained(checkpoint_path)