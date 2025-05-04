"""
Advanced audio processing utilities for voice cloning system
"""

import numpy as np
import torch
import torch.nn.functional as F
import librosa
from typing import List, Dict, Tuple, Optional, Union
import scipy.signal as signal

def normalize_audio(waveform: np.ndarray, target_db: float = -24.0) -> np.ndarray:
    """
    Normalize audio to target dB level
    
    Args:
        waveform: Audio waveform
        target_db: Target dB level
        
    Returns:
        Normalized waveform
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms > 0:
        # Convert target_db to linear gain
        target_amplitude = 10 ** (target_db / 20)
        # Apply gain to reach target
        gain = target_amplitude / rms
        normalized = waveform * gain
        return normalized
    else:
        # If silent, return as is
        return waveform

def extract_voice_characteristics(waveform: np.ndarray, sample_rate: int = 22050) -> Dict[str, float]:
    """
    Extract voice characteristics for speaker analysis
    
    Args:
        waveform: Audio waveform
        sample_rate: Audio sample rate
    
    Returns:
        Dictionary of voice characteristics
    """
    # Ensure minimum length for feature extraction
    if len(waveform) < sample_rate * 0.5:  # At least 0.5 seconds
        # Pad if too short
        waveform = np.pad(waveform, (0, int(sample_rate * 0.5) - len(waveform)))
    
    # Extract pitch (f0) using PYIN
    f0, voiced_flag, _ = librosa.pyin(
        waveform, 
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sample_rate,
        frame_length=2048
    )
    
    # Get valid pitch values
    valid_f0 = f0[~np.isnan(f0)]
    
    # Calculate pitch statistics
    pitch_mean = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0
    pitch_std = float(np.std(valid_f0)) if len(valid_f0) > 0 else 0.0
    pitch_range = float(np.ptp(valid_f0)) if len(valid_f0) > 0 else 0.0
    
    # Extract formants (vowel resonances)
    # This is a simplified approximation
    n_formants = 3
    formants = []
    
    # Simple FFT-based formant estimation
    try:
        # Get the power spectrum
        power_spec = np.abs(librosa.stft(waveform)) ** 2
        freqs = librosa.fft_frequencies(sr=sample_rate)
        
        # Average over time
        mean_spec = np.mean(power_spec, axis=1)
        
        # Smooth the spectrum
        smooth_spec = signal.savgol_filter(mean_spec, 11, 2)
        
        # Find peaks
        peaks, _ = signal.find_peaks(smooth_spec)
        peak_freqs = freqs[peaks]
        peak_mags = smooth_spec[peaks]
        
        # Sort by magnitude and get top n_formants
        sorted_idx = np.argsort(peak_mags)[::-1]
        top_freqs = peak_freqs[sorted_idx[:n_formants]]
        
        # Sort by frequency
        formants = sorted(top_freqs)[:n_formants]
    except:
        # Fallback values if extraction fails
        formants = [500, 1500, 2500]
    
    # Extract spectral centroid (brightness)
    centroid = float(np.mean(librosa.feature.spectral_centroid(
        y=waveform, sr=sample_rate)[0]))
    
    # Extract spectral contrast (voice timbre)
    contrast = float(np.mean(librosa.feature.spectral_contrast(
        y=waveform, sr=sample_rate)[0]))
    
    # Extract speech rate approximation using energy peaks
    rms = librosa.feature.rms(y=waveform)[0]
    peaks, _ = signal.find_peaks(rms, height=np.mean(rms))
    speech_rate = len(peaks) / (len(waveform) / sample_rate) if len(waveform) > 0 else 0
    
    return {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pitch_range": pitch_range,
        "formant1": float(formants[0]) if len(formants) > 0 else 0.0,
        "formant2": float(formants[1]) if len(formants) > 1 else 0.0,
        "formant3": float(formants[2]) if len(formants) > 2 else 0.0,
        "spectral_centroid": centroid,
        "spectral_contrast": contrast,
        "speech_rate": float(speech_rate)
    }

def match_target_amplitude(waveform: np.ndarray, target_db: float) -> np.ndarray:
    """
    Match audio amplitude to target dB
    
    Args:
        waveform: Audio waveform
        target_db: Target amplitude in dB
        
    Returns:
        Amplitude normalized waveform
    """
    rms = np.sqrt(np.mean(waveform ** 2))
    target_rms = 10 ** (target_db / 20)
    
    if rms > 0:
        gain = target_rms / rms
        return waveform * gain
    else:
        return waveform

def compute_melspectrogram_similarity(mel1: torch.Tensor, mel2: torch.Tensor) -> float:
    """
    Compute similarity between two mel spectrograms
    
    Args:
        mel1: First mel spectrogram [channels, time]
        mel2: Second mel spectrogram [channels, time]
        
    Returns:
        Similarity score (higher means more similar)
    """
    # Ensure same time dimension by padding/truncating
    if mel1.shape[1] != mel2.shape[1]:
        target_len = min(mel1.shape[1], mel2.shape[1])
        mel1 = mel1[:, :target_len]
        mel2 = mel2[:, :target_len]
    
    # Compute cosine similarity
    mel1_flat = mel1.reshape(-1)
    mel2_flat = mel2.reshape(-1)
    
    similarity = F.cosine_similarity(mel1_flat.unsqueeze(0), mel2_flat.unsqueeze(0))
    return similarity.item()

def enhance_voice_quality(waveform: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
    """
    Enhance voice quality with basic audio processing techniques
    
    Args:
        waveform: Audio waveform
        sample_rate: Audio sample rate
        
    Returns:
        Enhanced waveform
    """
    # High-pass filter to remove rumble
    b, a = signal.butter(4, 80/(sample_rate/2), 'highpass')
    waveform = signal.filtfilt(b, a, waveform)
    
    # Slight compression for consistent levels
    # Simple implementation of compression
    threshold = 0.5
    ratio = 4.0
    makeup_gain = 1.2
    
    # Apply compression
    compressed = np.zeros_like(waveform)
    for i, sample in enumerate(waveform):
        if abs(sample) > threshold:
            sign = 1 if sample > 0 else -1
            compressed[i] = sign * (threshold + (abs(sample) - threshold) / ratio)
        else:
            compressed[i] = sample
    
    # Apply makeup gain
    compressed = compressed * makeup_gain
    
    # Ensure we don't clip
    if np.max(np.abs(compressed)) > 0.99:
        compressed = compressed / np.max(np.abs(compressed)) * 0.99
    
    return compressed

def apply_dynamic_range_compression(
    waveform: np.ndarray, 
    threshold: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
    sample_rate: int = 22050
) -> np.ndarray:
    """
    Apply dynamic range compression to audio
    
    Args:
        waveform: Audio waveform
        threshold: Threshold in dB
        ratio: Compression ratio
        attack_ms: Attack time in ms
        release_ms: Release time in ms
        sample_rate: Sample rate
        
    Returns:
        Compressed waveform
    """
    # Convert threshold to linear
    threshold_linear = 10 ** (threshold / 20)
    
    # Compute time constants
    attack_samples = int(attack_ms * sample_rate / 1000)
    release_samples = int(release_ms * sample_rate / 1000)
    
    # Ensure at least 1 sample
    attack_samples = max(1, attack_samples)
    release_samples = max(1, release_samples)
    
    # Time constants for envelope follower
    attack_coef = np.exp(-1.0 / attack_samples)
    release_coef = np.exp(-1.0 / release_samples)
    
    # Envelope detection
    envelope = np.zeros_like(waveform)
    for i in range(len(waveform)):
        # Current sample amplitude
        amplitude = abs(waveform[i])
        
        # Determine whether to use attack or release time constant
        if i > 0:
            if amplitude > envelope[i-1]:
                envelope[i] = attack_coef * envelope[i-1] + (1 - attack_coef) * amplitude
            else:
                envelope[i] = release_coef * envelope[i-1] + (1 - release_coef) * amplitude
        else:
            envelope[i] = amplitude
    
    # Compute gain reduction
    gain_reduction = np.ones_like(envelope)
    mask = envelope > threshold_linear
    
    # Apply compressor formula: gain = (level/threshold)^(1/ratio-1)
    if np.any(mask):
        above_threshold = envelope[mask]
        gain_reduction[mask] = (above_threshold / threshold_linear) ** (1/ratio - 1)
    
    # Apply gain reduction to signal
    return waveform * gain_reduction