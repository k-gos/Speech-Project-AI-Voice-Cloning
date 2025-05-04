"""
Visualization utilities for voice cloning system
"""

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from pathlib import Path
import torch
from matplotlib.figure import Figure
import io
from typing import Tuple, List, Dict, Optional, Union
import os

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')


def plot_waveform(waveform: np.ndarray, 
                 sample_rate: int = 22050, 
                 title: str = "Waveform") -> Figure:
    """
    Plot audio waveform
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(waveform) / sample_rate, len(waveform)), waveform)
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_spectrogram(spectrogram: np.ndarray,
                    sample_rate: int = 22050,
                    hop_length: int = 256,
                    title: str = "Spectrogram") -> Figure:
    """
    Plot spectrogram
    
    Args:
        spectrogram: Spectrogram (or mel spectrogram)
        sample_rate: Sample rate
        hop_length: Hop length used for STFT
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(10, 4))
    
    # If input is magnitude spectrogram, convert to dB
    if np.max(spectrogram) > 2.0:  # Arbitrary threshold
        spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    
    librosa.display.specshow(
        spectrogram,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log" if spectrogram.shape[0] > 128 else "mel"  # Use log scale for STFT, mel scale for mel spec
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_attention_weights(attention_weights: np.ndarray,
                          title: str = "Attention Weights") -> Figure:
    """
    Plot attention weights matrix
    
    Args:
        attention_weights: Attention weights matrix
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar()
    plt.xlabel("Decoder timestep")
    plt.ylabel("Encoder timestep")
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_mel_spectrogram(mel_spectrogram: np.ndarray,
                        sample_rate: int = 22050,
                        hop_length: int = 256,
                        title: str = "Mel Spectrogram") -> Figure:
    """
    Plot mel spectrogram
    
    Args:
        mel_spectrogram: Mel spectrogram
        sample_rate: Sample rate
        hop_length: Hop length used for STFT
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(10, 4))
    
    # Convert to dB if needed
    if np.max(mel_spectrogram) > 2.0:
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
    librosa.display.specshow(
        mel_spectrogram,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel"
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_training_history(training_losses: List[float],
                         validation_losses: List[float],
                         title: str = "Training History") -> Figure:
    """
    Plot training and validation loss history
    
    Args:
        training_losses: List of training losses
        validation_losses: List of validation losses
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig


def plot_voice_embedding_tsne(embeddings: np.ndarray, 
                             labels: List[str],
                             title: str = "Voice Embedding t-SNE") -> Figure:
    """
    Plot t-SNE visualization of voice embeddings
    
    Args:
        embeddings: Voice embeddings
        labels: Speaker labels
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Perform t-SNE dimensionality reduction
    from sklearn.manifold import TSNE
    
    # Reduce to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create plot
    fig = plt.figure(figsize=(12, 10))
    
    # Get unique labels and assign colors
    unique_labels = list(set(labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each speaker group
    for i, label in enumerate(unique_labels):
        indices = [j for j, l in enumerate(labels) if l == label]
        plt.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            color=colors[i],
            label=label,
            alpha=0.7
        )
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_speaker_similarity(similarity_matrix: np.ndarray, 
                          speaker_ids: List[str],
                          title: str = "Speaker Similarity") -> Figure:
    """
    Plot speaker similarity matrix as a heatmap
    
    Args:
        similarity_matrix: Speaker similarity matrix
        speaker_ids: Speaker IDs
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar()
    
    # Add labels
    plt.xticks(range(len(speaker_ids)), speaker_ids, rotation=90)
    plt.yticks(range(len(speaker_ids)), speaker_ids)
    
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_pitch_contour(waveform: np.ndarray, 
                      sample_rate: int = 22050,
                      title: str = "Pitch Contour") -> Figure:
    """
    Plot pitch contour of audio
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Extract pitch using librosa
    f0, voiced_flag, voiced_probs = librosa.pyin(
        waveform, 
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sample_rate
    )
    
    # Create time array
    times = librosa.times_like(f0, sr=sample_rate)
    
    # Plot
    fig = plt.figure(figsize=(10, 6))
    plt.plot(times, f0, label='f0', color='blue', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    return fig


def generate_audio_report(audio_path: str, output_dir: str):
    """
    Generate a comprehensive audio analysis report
    
    Args:
        audio_path: Path to audio file
        output_dir: Output directory for report files
    """
    import os
    from pathlib import Path
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Generate plots
    
    # 1. Waveform
    fig_waveform = plot_waveform(y, sr, "Audio Waveform")
    fig_waveform.savefig(output_dir / "waveform.png")
    
    # 2. Spectrogram
    D = np.abs(librosa.stft(y))
    fig_spec = plot_spectrogram(D, sr, title="Spectrogram")
    fig_spec.savefig(output_dir / "spectrogram.png")
    
    # 3. Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    fig_mel = plot_mel_spectrogram(mel_spec, sr, title="Mel Spectrogram")
    fig_mel.savefig(output_dir / "mel_spectrogram.png")
    
    # 4. Pitch contour
    fig_pitch = plot_pitch_contour(y, sr, "Pitch Contour")
    fig_pitch.savefig(output_dir / "pitch_contour.png")
    
    # 5. MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    fig_mfcc = plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title("MFCC")
    plt.tight_layout()
    fig_mfcc.savefig(output_dir / "mfcc.png")
    
    # Generate text report with audio statistics
    audio_duration = librosa.get_duration(y=y, sr=sr)
    
    with open(output_dir / "audio_report.txt", "w") as f:
        f.write(f"Audio Analysis Report for {os.path.basename(audio_path)}\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Duration: {audio_duration:.2f} seconds\n")
        f.write(f"Sample Rate: {sr} Hz\n")
        f.write(f"Number of Samples: {len(y)}\n")
        f.write(f"Max Amplitude: {np.max(np.abs(y)):.4f}\n")
        f.write(f"RMS Energy: {np.sqrt(np.mean(y**2)):.4f}\n\n")
        
        # Add pitch statistics
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
        )
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            f.write("Pitch Statistics:\n")
            f.write(f"  Mean Pitch: {np.mean(valid_f0):.2f} Hz\n")
            f.write(f"  Min Pitch: {np.min(valid_f0):.2f} Hz\n")
            f.write(f"  Max Pitch: {np.max(valid_f0):.2f} Hz\n")
            f.write(f"  Pitch Range: {np.max(valid_f0) - np.min(valid_f0):.2f} Hz\n\n")
        
    print(f"Report generated in {output_dir}")


def figure_to_image(fig: Figure) -> np.ndarray:
    """
    Convert matplotlib figure to image array
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Image as numpy array
    """
    # Save figure to in-memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Load image from buffer
    from PIL import Image
    img = Image.open(buf)
    
    # Convert to numpy array
    return np.array(img)


if __name__ == "__main__":
    # Test the visualization utilities
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Visualization Utilities")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--output", type=str, default="visualization", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"Generating audio report for {args.audio}...")
    generate_audio_report(args.audio, args.output)