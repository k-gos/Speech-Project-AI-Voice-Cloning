"""
Audio Utility Module

Functions for audio processing, feature extraction, and manipulation.
"""
import os
import numpy as np
import torch
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from matplotlib.figure import Figure
from scipy.io.wavfile import write as write_wav


def load_audio(file_path: str, 
              target_sr: int = 22050,
              max_duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        max_duration: Maximum duration in seconds (optional)
        
    Returns:
        Tuple of (waveform, sample_rate)
    """
    try:
        # Load audio with librosa
        waveform, sr = librosa.load(file_path, sr=target_sr)
        
        # Trim to max duration if specified
        if max_duration is not None:
            max_samples = int(max_duration * target_sr)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]
                
        return waveform, target_sr
    
    except Exception as e:
        print(f"Error loading audio file {file_path}: {str(e)}")
        # Return empty array with target sample rate
        return np.zeros(target_sr), target_sr


def save_audio(waveform: Union[np.ndarray, torch.Tensor], 
              file_path: str, 
              sample_rate: int = 22050):
    """
    Save audio waveform to file
    
    Args:
        waveform: Audio waveform
        file_path: Output file path
        sample_rate: Sample rate
    """
    # Convert torch tensor to numpy if needed
    if torch.is_tensor(waveform):
        waveform = waveform.detach().cpu().numpy()
    
    # Make sure the directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Normalize if needed
    if np.abs(waveform).max() > 1.0:
        waveform = waveform / np.abs(waveform).max() * 0.9
        
    # Save as WAV file
    sf.write(file_path, waveform, sample_rate)


def extract_mel_spectrogram(waveform: np.ndarray, 
                           sample_rate: int = 22050,
                           n_fft: int = 1024,
                           hop_length: int = 256,
                           win_length: int = 1024,
                           n_mels: int = 80,
                           fmin: int = 0,
                           fmax: int = 8000,
                           normalize: bool = True) -> np.ndarray:
    """
    Extract mel spectrogram from audio waveform
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        n_fft: FFT size
        hop_length: Hop length for STFT
        win_length: Window length for STFT
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency
        normalize: Whether to normalize the spectrogram
        
    Returns:
        Mel spectrogram
    """
    # Extract mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    
    # Convert to log scale
    mel = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize if requested
    if normalize:
        mel = (mel - mel.min()) / (mel.max() - mel.min()) * 2 - 1
        
    return mel


def extract_audio_features(audio_path: str) -> Dict[str, float]:
    """
    Extract acoustic features from audio file
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary of audio features
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Extract features
    # Pitch
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )
    pitch_mean = np.nanmean(f0) if not np.isnan(np.nanmean(f0)) else 0
    
    # Energy
    energy = np.mean(librosa.feature.rms(y=y)[0])
    
    # Spectral centroid
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    
    # Zero-crossing rate (related to voice quality)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y)[0])
    
    # Spectral contrast (related to voice quality)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    
    # Normalize values to 0-1 range
    pitch_mean_norm = min(max(pitch_mean / 500, 0), 1)
    energy_norm = min(max(energy * 50, 0), 1)
    centroid_norm = min(max(centroid / 5000, 0), 1) 
    zcr_norm = min(max(zcr * 10, 0), 1)
    contrast_norm = min(max(contrast / 50 + 0.5, 0), 1)
    
    return {
        "pitch_mean": pitch_mean_norm,
        "energy": energy_norm,
        "spectral_centroid": centroid_norm,
        "zero_crossing_rate": zcr_norm,
        "spectral_contrast": contrast_norm
    }


def extract_pitch(waveform: np.ndarray, 
                 sample_rate: int = 22050, 
                 hop_length: int = 256) -> np.ndarray:
    """
    Extract pitch (f0) contour from waveform
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        hop_length: Hop length for pitch extraction
        
    Returns:
        Pitch contour
    """
    # Extract pitch using PYIN algorithm
    f0, voiced_flag, voiced_probs = librosa.pyin(
        waveform, 
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sample_rate,
        hop_length=hop_length
    )
    
    # Replace NaN values with 0
    f0 = np.nan_to_num(f0)
    
    return f0


def extract_energy(waveform: np.ndarray, 
                  hop_length: int = 256) -> np.ndarray:
    """
    Extract energy contour from waveform
    
    Args:
        waveform: Audio waveform
        hop_length: Hop length for energy extraction
        
    Returns:
        Energy contour
    """
    # Extract RMS energy
    energy = librosa.feature.rms(y=waveform, hop_length=hop_length)[0]
    
    return energy


def adjust_pitch(waveform: np.ndarray, 
                sample_rate: int = 22050,
                n_steps: float = 0.0) -> np.ndarray:
    """
    Adjust pitch of waveform
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        n_steps: Number of semitones to shift (positive = up, negative = down)
        
    Returns:
        Pitch-adjusted waveform
    """
    return librosa.effects.pitch_shift(waveform, sr=sample_rate, n_steps=n_steps)


def adjust_tempo(waveform: np.ndarray, 
                rate: float = 1.0) -> np.ndarray:
    """
    Adjust tempo of waveform
    
    Args:
        waveform: Audio waveform
        rate: Time stretch factor (>1 = faster, <1 = slower)
        
    Returns:
        Tempo-adjusted waveform
    """
    return librosa.effects.time_stretch(waveform, rate=rate)


def apply_emotion_to_waveform(waveform: np.ndarray, 
                             sample_rate: int = 22050, 
                             emotion: str = "neutral") -> np.ndarray:
    """
    Apply emotion modification to waveform
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        emotion: Target emotion
        
    Returns:
        Modified waveform
    """
    # Simple emotion modifications (in a real system, this would be more sophisticated)
    if emotion == "happy":
        # Increase pitch and speed slightly
        waveform = librosa.effects.pitch_shift(waveform, sr=sample_rate, n_steps=1)
        waveform = librosa.effects.time_stretch(waveform, rate=1.1)
    elif emotion == "sad":
        # Lower pitch and slow down
        waveform = librosa.effects.pitch_shift(waveform, sr=sample_rate, n_steps=-1)
        waveform = librosa.effects.time_stretch(waveform, rate=0.9)
    elif emotion == "angry":
        # Increase energy and add slight distortion
        waveform = waveform * 1.2
        waveform = np.clip(waveform, -1.0, 1.0)  # Clip to prevent extreme distortion
    elif emotion == "surprised":
        # Higher pitch, faster at beginning
        waveform = librosa.effects.pitch_shift(waveform, sr=sample_rate, n_steps=2)
        # Create emphasis on beginning
        env = np.linspace(1.2, 1.0, len(waveform))
        waveform = waveform * env
    elif emotion == "fear":
        # Add slight tremolo effect
        tremolo_rate = 6.0  # Hz
        tremolo_depth = 0.2
        t = np.arange(len(waveform)) / sample_rate
        tremolo = 1.0 + tremolo_depth * np.sin(2.0 * np.pi * tremolo_rate * t)
        waveform = waveform * tremolo
        
    # Normalize after modifications
    if np.abs(waveform).max() > 0:
        waveform = waveform / np.abs(waveform).max() * 0.9
        
    return waveform


def visualize_audio(waveform: np.ndarray, 
                   sample_rate: int = 22050,
                   title: str = "Audio Waveform") -> Figure:
    """
    Visualize audio waveform and spectrogram
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(waveform, sr=sample_rate)
    plt.title(f"{title} - Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{title} - Spectrogram")
    
    plt.tight_layout()
    return fig


def compare_audio(original_waveform: np.ndarray, 
                 generated_waveform: np.ndarray,
                 sample_rate: int = 22050,
                 title: str = "Audio Comparison") -> Figure:
    """
    Compare original and generated audio waveforms
    
    Args:
        original_waveform: Original audio waveform
        generated_waveform: Generated audio waveform
        sample_rate: Sample rate
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Make sure both waveforms have the same length for comparison
    min_length = min(len(original_waveform), len(generated_waveform))
    original_waveform = original_waveform[:min_length]
    generated_waveform = generated_waveform[:min_length]
    
    fig = plt.figure(figsize=(12, 10))
    
    # Plot original waveform
    plt.subplot(4, 1, 1)
    librosa.display.waveshow(original_waveform, sr=sample_rate)
    plt.title("Original Waveform")
    
    # Plot generated waveform
    plt.subplot(4, 1, 2)
    librosa.display.waveshow(generated_waveform, sr=sample_rate)
    plt.title("Generated Waveform")
    
    # Plot original spectrogram
    plt.subplot(4, 1, 3)
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original_waveform)), ref=np.max)
    librosa.display.specshow(D_orig, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Original Spectrogram")
    
    # Plot generated spectrogram
    plt.subplot(4, 1, 4)
    D_gen = librosa.amplitude_to_db(np.abs(librosa.stft(generated_waveform)), ref=np.max)
    librosa.display.specshow(D_gen, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Generated Spectrogram")
    
    plt.tight_layout()
    return fig


def extract_mfcc(waveform: np.ndarray, 
                sample_rate: int = 22050,
                n_mfcc: int = 13) -> np.ndarray:
    """
    Extract MFCC features from waveform
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        MFCC features
    """
    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc)
    return mfccs


def trim_silence(waveform: np.ndarray,
                top_db: float = 20.0) -> np.ndarray:
    """
    Trim leading and trailing silence from waveform
    
    Args:
        waveform: Audio waveform
        top_db: Threshold in decibels
        
    Returns:
        Trimmed waveform
    """
    trimmed_waveform, _ = librosa.effects.trim(waveform, top_db=top_db)
    return trimmed_waveform