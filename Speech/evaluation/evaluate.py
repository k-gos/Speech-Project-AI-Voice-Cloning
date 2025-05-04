"""
Evaluation script for voice cloning system
This is a standalone script to evaluate model performance.
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm
import datetime
import sys
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import project modules
from models.model import VoiceCloningModel
from models.speaker_encoder import SpeakerEncoder
from src.utils.audio import load_audio, save_audio, compare_audio
from src.utils.audio_helpers import extract_voice_characteristics, compute_melspectrogram_similarity
from src.utils.visualization import plot_speaker_similarity


class VoiceCloningEvaluator:
    """Full evaluator for voice cloning system"""
    
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
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Support different checkpoint formats
        if "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            model_state_dict = checkpoint["model"]
        else:
            model_state_dict = checkpoint
            
        # Initialize model
        self.model = VoiceCloningModel()
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize speaker encoder
        self.speaker_encoder = self.model.speaker_encoder
        
        print("Model loaded successfully")
        
        # Timestamp for report
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    def evaluate_speaker_similarity(self, 
                                   reference_audio_dir: str,
                                   synthesized_audio_dir: str):
        """
        Evaluate speaker similarity between reference and synthesized audio
        
        Args:
            reference_audio_dir: Directory with reference audio files
            synthesized_audio_dir: Directory with synthesized audio files
            
        Returns:
            Dictionary of similarity results
        """
        print("Evaluating speaker similarity...")
        
        # Find matching reference and synthesized files
        ref_files = sorted(list(Path(reference_audio_dir).glob("*.wav")))
        syn_files = sorted(list(Path(synthesized_audio_dir).glob("*.wav")))
        
        if not ref_files:
            print(f"No reference audio files found in {reference_audio_dir}")
            return {}
            
        if not syn_files:
            print(f"No synthesized audio files found in {synthesized_audio_dir}")
            return {}
        
        # Extract speaker embeddings
        ref_embeddings = []
        syn_embeddings = []
        
        print("Extracting reference embeddings...")
        for ref_file in tqdm(ref_files):
            waveform, sr = load_audio(str(ref_file))
            waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(self.device)
            embedding = self.speaker_encoder(waveform_tensor).squeeze().cpu().numpy()
            ref_embeddings.append(embedding)
        
        print("Extracting synthesized embeddings...")
        for syn_file in tqdm(syn_files):
            waveform, sr = load_audio(str(syn_file))
            waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(self.device)
            embedding = self.speaker_encoder(waveform_tensor).squeeze().cpu().numpy()
            syn_embeddings.append(embedding)
        
        # Convert to numpy arrays
        ref_embeddings = np.array(ref_embeddings)
        syn_embeddings = np.array(syn_embeddings)
        
        # Calculate similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
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
        output_file = self.output_dir / f"speaker_similarity_{self.timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
            
        # Plot similarity matrix
        fig = plot_speaker_similarity(
            similarity_matrix, 
            [f.stem for f in ref_files[:similarity_matrix.shape[0]]], 
            f"Speaker Similarity (Mean: {mean_similarity:.4f})"
        )
        
        fig.savefig(self.output_dir / f"speaker_similarity_matrix_{self.timestamp}.png")
        
        print(f"Speaker similarity results:")
        print(f"  Mean: {mean_similarity:.4f}")
        print(f"  Max: {results['max_similarity']:.4f}")
        print(f"  Min: {results['min_similarity']:.4f}")
        print(f"Results saved to {output_file}")
        
        return results
    
    def evaluate_mel_cepstral_distortion(self,
                                        reference_audio_dir: str,
                                        synthesized_audio_dir: str):
        """
        Calculate Mel Cepstral Distortion between reference and synthesized audio
        
        Args:
            reference_audio_dir: Directory with reference audio files
            synthesized_audio_dir: Directory with synthesized audio files
            
        Returns:
            Dictionary of MCD results
        """
        print("Evaluating Mel Cepstral Distortion...")
        
        # Find matching files by name
        ref_path = Path(reference_audio_dir)
        syn_path = Path(synthesized_audio_dir)
        
        # Get all reference files
        ref_files = list(ref_path.glob("*.wav"))
        
        if not ref_files:
            print(f"No reference audio files found in {reference_audio_dir}")
            return {}
        
        # Calculate MCD for each pair
        mcd_scores = []
        file_pairs = []
        
        for ref_file in tqdm(ref_files):
            # Find matching synthesized file
            syn_file = syn_path / ref_file.name
            
            if not syn_file.exists():
                print(f"No matching synthesized file for {ref_file.name}")
                continue
            
            # Load audio files
            ref_wav, ref_sr = load_audio(str(ref_file))
            syn_wav, syn_sr = load_audio(str(syn_file))
            
            # Extract MFCCs
            ref_mfcc = librosa.feature.mfcc(y=ref_wav, sr=ref_sr, n_mfcc=13)
            syn_mfcc = librosa.feature.mfcc(y=syn_wav, sr=syn_sr, n_mfcc=13)
            
            # Dynamic time warping to align sequences
            from fastdtw import fastdtw
            from scipy.spatial.distance import euclidean
            
            # Transpose MFCCs to have time as first dimension
            ref_mfcc = ref_mfcc.T
            syn_mfcc = syn_mfcc.T
            
            # Calculate DTW distance
            distance, _ = fastdtw(ref_mfcc, syn_mfcc, dist=euclidean)
            
            # Normalize by sequence length
            mcd = distance / max(len(ref_mfcc), len(syn_mfcc))
            
            mcd_scores.append(mcd)
            file_pairs.append((str(ref_file), str(syn_file)))
        
        # Calculate statistics
        if mcd_scores:
            mean_mcd = np.mean(mcd_scores)
            std_mcd = np.std(mcd_scores)
            min_mcd = np.min(mcd_scores)
            max_mcd = np.max(mcd_scores)
        else:
            mean_mcd = std_mcd = min_mcd = max_mcd = 0.0
        
        # Log results
        results = {
            "mean_mcd": float(mean_mcd),
            "std_mcd": float(std_mcd),
            "min_mcd": float(min_mcd),
            "max_mcd": float(max_mcd),
            "file_pairs": file_pairs,
            "mcd_scores": [float(score) for score in mcd_scores]
        }
        
        # Save results
        output_file = self.output_dir / f"mcd_results_{self.timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot histogram of MCD scores
        plt.figure(figsize=(10, 6))
        plt.hist(mcd_scores, bins=20, alpha=0.7)
        plt.axvline(mean_mcd, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_mcd:.2f}')
        plt.title('Mel Cepstral Distortion Scores')
        plt.xlabel('MCD')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"mcd_histogram_{self.timestamp}.png")
        
        print(f"MCD Evaluation results:")
        print(f"  Mean MCD: {mean_mcd:.2f}")
        print(f"  Min MCD: {min_mcd:.2f}")
        print(f"  Max MCD: {max_mcd:.2f}")
        print(f"Results saved to {output_file}")
        
        return results
    
    def evaluate_voice_characteristics(self,
                                      reference_audio_dir: str,
                                      synthesized_audio_dir: str):
        """
        Compare voice characteristics between reference and synthesized audio
        
        Args:
            reference_audio_dir: Directory with reference audio files
            synthesized_audio_dir: Directory with synthesized audio files
            
        Returns:
            Dictionary of voice characteristics comparison results
        """
        print("Evaluating voice characteristics preservation...")
        
        # Find matching files by name
        ref_path = Path(reference_audio_dir)
        syn_path = Path(synthesized_audio_dir)
        
        # Get all reference files
        ref_files = list(ref_path.glob("*.wav"))
        
        if not ref_files:
            print(f"No reference audio files found in {reference_audio_dir}")
            return {}
        
        # Calculate characteristics for each pair
        characteristics_diffs = []
        file_pairs = []
        
        for ref_file in tqdm(ref_files):
            # Find matching synthesized file
            syn_file = syn_path / ref_file.name
            
            if not syn_file.exists():
                print(f"No matching synthesized file for {ref_file.name}")
                continue
            
            # Load audio files
            ref_wav, ref_sr = load_audio(str(ref_file))
            syn_wav, syn_sr = load_audio(str(syn_file))
            
            # Extract voice characteristics
            ref_chars = extract_voice_characteristics(ref_wav, ref_sr)
            syn_chars = extract_voice_characteristics(syn_wav, syn_sr)
            
            # Calculate differences
            diff = {}
            for key in ref_chars:
                if key in syn_chars:
                    diff[key] = abs(ref_chars[key] - syn_chars[key])
            
            characteristics_diffs.append(diff)
            file_pairs.append((str(ref_file), str(syn_file)))
        
        # Calculate statistics
        if characteristics_diffs:
            # Average differences for each characteristic
            avg_diffs = {}
            for key in characteristics_diffs[0]:
                values = [d.get(key, 0.0) for d in characteristics_diffs]
                avg_diffs[key] = float(np.mean(values))
            
            # Overall average difference
            all_values = []
            for diff in characteristics_diffs:
                all_values.extend(list(diff.values()))
            overall_avg = float(np.mean(all_values))
        else:
            avg_diffs = {}
            overall_avg = 0.0
        
        # Log results
        results = {
            "overall_average_difference": overall_avg,
            "characteristic_differences": avg_diffs,
            "file_pairs": file_pairs
        }
        
        # Save results
        output_file = self.output_dir / f"voice_characteristics_{self.timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot bar chart of characteristic differences
        plt.figure(figsize=(12, 6))
        plt.bar(avg_diffs.keys(), avg_diffs.values())
        plt.title('Voice Characteristic Differences')
        plt.xlabel('Characteristic')
        plt.ylabel('Absolute Difference')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"voice_characteristics_diff_{self.timestamp}.png")
        
        print(f"Voice Characteristics Evaluation results:")
        print(f"  Overall Average Difference: {overall_avg:.4f}")
        for key, value in avg_diffs.items():
            print(f"  {key}: {value:.4f}")
        print(f"Results saved to {output_file}")
        
        return results
    
    def evaluate_with_target_text(self, 
                                 reference_audio_dir: str,
                                 target_texts: Dict[str, str],
                                 emotions: List[str] = None):
        """
        Evaluate model by generating speech for target texts
        
        Args:
            reference_audio_dir: Directory with reference audio files
            target_texts: Dictionary mapping file names to target texts
            emotions: List of emotions to evaluate (default: neutral)
            
        Returns:
            Dictionary of evaluation results
        """
        if emotions is None:
            emotions = ["neutral"]
        
        print("Evaluating model with target texts...")
        
        # Output directory for generated samples
        samples_dir = self.output_dir / f"generated_samples_{self.timestamp}"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Find reference files
        ref_files = list(Path(reference_audio_dir).glob("*.wav"))
        
        if not ref_files:
            print(f"No reference audio files found in {reference_audio_dir}")
            return {}
        
        # Generate samples
        generated_files = []
        
        with torch.no_grad():
            for ref_file in tqdm(ref_files):
                # Check if we have target text for this file
                if ref_file.stem in target_texts:
                    text = target_texts[ref_file.stem]
                else:
                    # Skip if no target text
                    continue
                
                # Generate for each emotion
                for emotion in emotions:
                    # Output file path
                    output_file = samples_dir / f"{ref_file.stem}_{emotion}.wav"
                    
                    # Generate speech
                    try:
                        # Load reference audio
                        waveform, sr = load_audio(str(ref_file))
                        waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(self.device)
                        
                        # Extract speaker embedding
                        speaker_embedding = self.speaker_encoder(waveform_tensor)
                        
                        # Generate speech
                        outputs = self.model.forward(
                            text=text,
                            speaker_embedding=speaker_embedding,
                            emotion=emotion
                        )
                        
                        # Save generated speech
                        waveform = outputs["waveform"].squeeze().cpu().numpy()
                        save_audio(waveform, str(output_file))
                        
                        generated_files.append({
                            "reference": str(ref_file),
                            "text": text,
                            "emotion": emotion,
                            "output": str(output_file)
                        })
                    except Exception as e:
                        print(f"Error generating speech for {ref_file.stem}, emotion {emotion}: {str(e)}")
        
        # Save sample information
        output_file = samples_dir / "sample_info.json"
        with open(output_file, "w") as f:
            json.dump(generated_files, f, indent=2)
            
        print(f"Generated {len(generated_files)} samples in {samples_dir}")
        
        # Return sample information
        return {
            "samples_dir": str(samples_dir),
            "generated_files": generated_files
        }
    
    def generate_evaluation_report(self):
        """
        Generate comprehensive evaluation report
        """
        # Report file path
        report_path = self.output_dir / f"evaluation_report_{self.timestamp}.md"
        
        # Create report content
        report = [
            f"# Voice Cloning Evaluation Report",
            f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model Information",
            f"- Device: {self.device}",
            "",
            "## Evaluation Results",
            "",
            "### Speaker Similarity",
            "Measures how well the model preserves speaker identity.",
            "See `speaker_similarity_*.json` for detailed results.",
            "",
            "### Mel Cepstral Distortion",
            "Measures the acoustic difference between generated and reference speech.",
            "See `mcd_results_*.json` for detailed results.",
            "",
            "### Voice Characteristics",
            "Measures how well the model preserves specific voice characteristics.",
            "See `voice_characteristics_*.json` for detailed results.",
            "",
            "### Generated Samples",
            "Listen to samples in the `generated_samples_*` directory.",
            "",
            "## Conclusion",
            "This evaluation report provides objective metrics for voice cloning quality.",
            "For best assessment, combine these metrics with subjective listening tests.",
            "",
            "## References",
            "- MCD: Kubichek, R. (1993). Mel-cepstral distance measure for objective speech quality assessment.",
            "- Speaker Similarity: based on cosine distance between speaker embeddings.",
            ""
        ]
        
        # Write report to file
        with open(report_path, "w") as f:
            f.write("\n".join(report))
            
        print(f"Evaluation report saved to {report_path}")
        
        return str(report_path)
    
    def run_complete_evaluation(self, 
                              reference_dir: str, 
                              synthesis_dir: str,
                              target_texts: Optional[Dict[str, str]] = None,
                              emotions: Optional[List[str]] = None):
        """
        Run complete evaluation suite
        
        Args:
            reference_dir: Directory with reference audio files
            synthesis_dir: Directory with synthesized audio files or to save new ones
            target_texts: Dictionary of target texts for new synthesis
            emotions: List of emotions to evaluate
            
        Returns:
            Dictionary of all evaluation results
        """
        results = {}
        
        # Speaker similarity
        if Path(synthesis_dir).exists():
            print("\n=== Evaluating Speaker Similarity ===")
            sim_results = self.evaluate_speaker_similarity(reference_dir, synthesis_dir)
            results["speaker_similarity"] = sim_results
            
            print("\n=== Evaluating Mel Cepstral Distortion ===")
            mcd_results = self.evaluate_mel_cepstral_distortion(reference_dir, synthesis_dir)
            results["mcd"] = mcd_results
            
            print("\n=== Evaluating Voice Characteristics ===")
            vc_results = self.evaluate_voice_characteristics(reference_dir, synthesis_dir)
            results["voice_characteristics"] = vc_results
        
        # Target text synthesis
        if target_texts is not None:
            print("\n=== Generating Samples from Target Texts ===")
            synthesis_results = self.evaluate_with_target_text(
                reference_dir, target_texts, emotions
            )
            results["synthesis"] = synthesis_results
        
        # Generate report
        print("\n=== Generating Evaluation Report ===")
        report_path = self.generate_evaluation_report()
        results["report_path"] = report_path
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate voice cloning model")
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--reference_dir", type=str, required=True,
                       help="Directory with reference audio files")
    parser.add_argument("--synthesis_dir", type=str, default=None,
                       help="Directory with synthesized audio files or to save new ones")
    parser.add_argument("--target_texts", type=str, default=None,
                       help="JSON file mapping file names to target texts")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for evaluation results")
    parser.add_argument("--emotions", type=str, default="neutral",
                       help="Comma-separated list of emotions to evaluate")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cpu or cuda)")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse emotions
    emotions = args.emotions.split(",")
    
    # Load target texts if provided
    target_texts = None
    if args.target_texts:
        with open(args.target_texts, "r") as f:
            target_texts = json.load(f)
    
    # Initialize evaluator
    evaluator = VoiceCloningEvaluator(
        model_path=args.model,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run evaluation
    evaluator.run_complete_evaluation(
        reference_dir=args.reference_dir,
        synthesis_dir=args.synthesis_dir if args.synthesis_dir else output_dir / "synthesized",
        target_texts=target_texts,
        emotions=emotions
    )
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()