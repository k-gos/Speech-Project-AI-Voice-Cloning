"""
Evaluation script for voice cloning system
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import project modules
from models.model import VoiceCloningModel
from models.speaker_encoder import SpeakerEncoder
from utils.audio import load_audio, save_audio, extract_mel_spectrogram
from utils.visualization import plot_speaker_similarity


class VoiceCloningEvaluator:
    """Evaluator for voice cloning models"""
    
    def __init__(self, 
                model_path: str,
                output_dir: str,
                device: str = None):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to model checkpoint
            output_dir: Directory for evaluation outputs
            device: Device to use (cpu or cuda)
        """
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model, _ = VoiceCloningModel.load_checkpoint(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize speaker encoder for similarity measurements
        self.speaker_encoder = self.model.speaker_encoder
    
    def evaluate_speaker_similarity(self, 
                                   reference_audio_dir: str,
                                   synthesized_audio_dir: str):
        """
        Evaluate speaker similarity between reference and synthesized audio
        
        Args:
            reference_audio_dir: Directory with reference audio files
            synthesized_audio_dir: Directory with synthesized audio files
        """
        print("Evaluating speaker similarity...")
        
        # Find matching reference and synthesized files
        ref_files = sorted(list(Path(reference_audio_dir).glob("*.wav")))
        syn_files = sorted(list(Path(synthesized_audio_dir).glob("*.wav")))
        
        if not ref_files:
            print(f"No reference audio files found in {reference_audio_dir}")
            return
            
        if not syn_files:
            print(f"No synthesized audio files found in {synthesized_audio_dir}")
            return
        
        # Extract speaker embeddings
        ref_embeddings = []
        syn_embeddings = []
        
        print("Extracting reference embeddings...")
        for ref_file in tqdm(ref_files):
            embedding = self.speaker_encoder.extract_embedding(str(ref_file))
            ref_embeddings.append(embedding.squeeze().cpu().numpy())
        
        print("Extracting synthesized embeddings...")
        for syn_file in tqdm(syn_files):
            embedding = self.speaker_encoder.extract_embedding(str(syn_file))
            syn_embeddings.append(embedding.squeeze().cpu().numpy())
        
        # Convert to numpy arrays
        ref_embeddings = np.array(ref_embeddings)
        syn_embeddings = np.array(syn_embeddings)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(ref_embeddings, syn_embeddings)
        
        # Calculate diagonal (matching pairs) similarities
        n_pairs = min(len(ref_files), len(syn_files))
        diag_similarities = np.diag(similarity_matrix[:n_pairs, :n_pairs])
        
        # Calculate mean similarity
        mean_similarity = np.mean(diag_similarities)
        
        # Log results
        results = {
            "mean_similarity": float(mean_similarity),
            "max_similarity": float(np.max(diag_similarities)),
            "min_similarity": float(np.min(diag_similarities)),
            "std_similarity": float(np.std(diag_similarities))
        }
        
        # Save results
        output_file = self.output_dir / "speaker_similarity.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
            
        # Plot similarity matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap="viridis")
        plt.colorbar()
        plt.xlabel("Synthesized Audio")
        plt.ylabel("Reference Audio")
        plt.title(f"Speaker Similarity (Mean: {mean_similarity:.4f})")
        plt.savefig(self.output_dir / "speaker_similarity_matrix.png")
        
        print(f"Speaker similarity results:")
        print(f"  Mean: {mean_similarity:.4f}")
        print(f"  Max: {results['max_similarity']:.4f}")
        print(f"  Min: {results['min_similarity']:.4f}")
        print(f"Results saved to {output_file}")
    
    def evaluate_mel_cepstral_distortion(self,
                                        reference_audio_dir: str,
                                        synthesized_audio_dir: str):
        """
        Calculate Mel Cepstral Distortion between reference and synthesized audio
        
        Args:
            reference_audio_dir: Directory with reference audio files
            synthesized_audio_dir: Directory with synthesized audio files
        """
        print("Evaluating Mel Cepstral Distortion...")
        
        # Find matching files by name
        ref_path = Path(reference_audio_dir)
        syn_path = Path(synthesized_audio_dir)
        
        # Get all reference files
        ref_files = list(ref_path.glob("*.wav"))
        
        if not ref_files:
            print(f"No reference audio files found in {reference_audio_dir}")
            return
        
        # Calculate MCD for each pair
        mcd_scores = []
        file_pairs = []
        
        for ref_file in tqdm(ref_files):
            # Find matching synthesized file
            syn_file = syn_path / ref_file.name
            
            if not syn_file.exists():
                print(f"No matching synthesized file for {ref_file.name}")
                continue
"""
Evaluation script for voice cloning system
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import project modules
from models.model import VoiceCloningModel
from models.speaker_encoder import SpeakerEncoder
from utils.audio import load_audio, save_audio, extract_mel_spectrogram
from utils.visualization import plot_speaker_similarity


class VoiceCloningEvaluator:
    """Evaluator for voice cloning models"""
    
    def __init__(self, 
                model_path: str,
                output_dir: str,
                device: str = None):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to model checkpoint
            output_dir: Directory for evaluation outputs
            device: Device to use (cpu or cuda)
        """
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model, _ = VoiceCloningModel.load_checkpoint(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize speaker encoder for similarity measurements
        self.speaker_encoder = self.model.speaker_encoder
    
    def evaluate_speaker_similarity(self, 
                                   reference_audio_dir: str,
                                   synthesized_audio_dir: str):
        """
        Evaluate speaker similarity between reference and synthesized audio
        
        Args:
            reference_audio_dir: Directory with reference audio files
            synthesized_audio_dir: Directory with synthesized audio files
        """
        print("Evaluating speaker similarity...")
        
        # Find matching reference and synthesized files
        ref_files = sorted(list(Path(reference_audio_dir).glob("*.wav")))
        syn_files = sorted(list(Path(synthesized_audio_dir).glob("*.wav")))
        
        if not ref_files:
            print(f"No reference audio files found in {reference_audio_dir}")
            return
            
        if not syn_files:
            print(f"No synthesized audio files found in {synthesized_audio_dir}")
            return
        
        # Extract speaker embeddings
        ref_embeddings = []
        syn_embeddings = []
        
        print("Extracting reference embeddings...")
        for ref_file in tqdm(ref_files):
            embedding = self.speaker_encoder.extract_embedding(str(ref_file))
            ref_embeddings.append(embedding.squeeze().cpu().numpy())
        
        print("Extracting synthesized embeddings...")
        for syn_file in tqdm(syn_files):
            embedding = self.speaker_encoder.extract_embedding(str(syn_file))
            syn_embeddings.append(embedding.squeeze().cpu().numpy())
        
        # Convert to numpy arrays
        ref_embeddings = np.array(ref_embeddings)
        syn_embeddings = np.array(syn_embeddings)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(ref_embeddings, syn_embeddings)
        
        # Calculate diagonal (matching pairs) similarities
        n_pairs = min(len(ref_files), len(syn_files))
        diag_similarities = np.diag(similarity_matrix[:n_pairs, :n_pairs])
        
        # Calculate mean similarity
        mean_similarity = np.mean(diag_similarities)
        
        # Log results
        results = {
            "mean_similarity": float(mean_similarity),
            "max_similarity": float(np.max(diag_similarities)),
            "min_similarity": float(np.min(diag_similarities)),
            "std_similarity": float(np.std(diag_similarities))
        }
        
        # Save results
        output_file = self.output_dir / "speaker_similarity.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
            
        # Plot similarity matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap="viridis")
        plt.colorbar()
        plt.xlabel("Synthesized Audio")
        plt.ylabel("Reference Audio")
        plt.title(f"Speaker Similarity (Mean: {mean_similarity:.4f})")
        plt.savefig(self.output_dir / "speaker_similarity_matrix.png")
        
        print(f"Speaker similarity results:")
        print(f"  Mean: {mean_similarity:.4f}")
        print(f"  Max: {results['max_similarity']:.4f}")
        print(f"  Min: {results['min_similarity']:.4f}")
        print(f"Results saved to {output_file}")
    
    def evaluate_mel_cepstral_distortion(self,
                                        reference_audio_dir: str,
                                        synthesized_audio_dir: str):
        """
        Calculate Mel Cepstral Distortion between reference and synthesized audio
        
        Args:
            reference_audio_dir: Directory with reference audio files
            synthesized_audio_dir: Directory with synthesized audio files
        """
        print("Evaluating Mel Cepstral Distortion...")
        
        # Find matching files by name
        ref_path = Path(reference_audio_dir)
        syn_path = Path(synthesized_audio_dir)
        
        # Get all reference files
        ref_files = list(ref_path.glob("*.wav"))
        
        if not ref_files:
            print(f"No reference audio files found in {reference_audio_dir}")
            return
        
        # Calculate MCD for each pair
        mcd_scores = []
        file_pairs = []
        
        for ref_file in tqdm(ref_files):
            # Find matching synthesized file
            syn_file = syn_path / ref_file.name
            
            if not syn_file.exists():
                print(f"No matching synthesized file for {ref_file.name}")
                continue