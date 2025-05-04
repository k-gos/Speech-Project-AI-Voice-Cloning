"""
Inference script for voice cloning system
Takes text and reference audio, generates speech in the reference voice.
"""

import torch
import numpy as np
import argparse
import os
import time
from pathlib import Path
import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import project modules
from models.model import VoiceCloningModel
from utils.audio import load_audio, save_audio, apply_emotion_to_waveform, visualize_audio
from utils.text import prepare_text_for_tts, clean_text_for_filename


class VoiceCloningInference:
    """Inference class for voice cloning"""
    
    def __init__(self, 
                checkpoint_path: str, 
                device: str = None,
                use_emotion_effects: bool = True):
        """
        Initialize inference module
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use (cpu or cuda)
            use_emotion_effects: Whether to apply additional emotion effects to output
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model, _ = VoiceCloningModel.load_checkpoint(checkpoint_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        
        # Additional settings
        self.use_emotion_effects = use_emotion_effects
        
        print("Model loaded successfully.")
    
    def generate_speech(self, 
                       text: str, 
                       reference_audio: str,
                       output_path: str = None,
                       emotion: str = "neutral",
                       emotion_reference: str = None,
                       cache_speaker: bool = True) -> str:
        """
        Generate speech from text using reference voice
        
        Args:
            text: Input text
            reference_audio: Path to reference audio file
            output_path: Path to save generated audio
            emotion: Target emotion
            emotion_reference: Path to emotion reference audio
            cache_speaker: Whether to cache speaker embedding
            
        Returns:
            Path to generated audio file
        """
        # Prepare text
        print(f"Processing text: '{text}'")
        
        # Extract speaker embedding
        if hasattr(self, 'cached_speaker_embedding') and cache_speaker:
            speaker_embedding = self.cached_speaker_embedding
            print("Using cached speaker embedding")
        else:
            print(f"Extracting speaker embedding from {reference_audio}...")
            speaker_embedding = self.model.get_speaker_embedding(reference_audio)
            
            # Cache for future use
            if cache_speaker:
                self.cached_speaker_embedding = speaker_embedding
        
        # Generate speech with model
        print("Generating speech...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.forward(
                text=text,
                speaker_embedding=speaker_embedding,
                emotion=emotion,
                emotion_reference=emotion_reference
            )
            
            waveform = outputs["waveform"]
        
        gen_time = time.time() - start_time
        audio_duration = waveform.size(-1) / self.model.config["sampling_rate"]
        rtf = gen_time / audio_duration
        print(f"Audio generated in {gen_time:.2f}s (Duration: {audio_duration:.2f}s, RTF: {rtf:.2f})")
        
        # Apply additional emotion effects if needed
        if self.use_emotion_effects and emotion != "neutral":
            waveform_np = waveform.squeeze().cpu().numpy()
            waveform_np = apply_emotion_to_waveform(
                waveform_np, 
                sample_rate=self.model.config["sampling_rate"],
                emotion=emotion
            )
        else:
            waveform_np = waveform.squeeze().cpu().numpy()
        
        # Save output
        if output_path is None:
            # Create filename from text
            clean_name = clean_text_for_filename(text)
            output_path = f"output_{clean_name}_{emotion}.wav"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save audio
        save_audio(
            waveform_np, 
            file_path=output_path, 
            sample_rate=self.model.config["sampling_rate"]
        )
        
        print(f"Audio saved to {output_path}")
        return output_path
    
    def generate_speech_from_long_text(self, 
                                      text: str, 
                                      reference_audio: str,
                                      output_dir: str,
                                      emotion: str = "neutral",
                                      combine_output: bool = True) -> str:
        """
        Generate speech from long text by splitting into chunks
        
        Args:
            text: Long text input
            reference_audio: Path to reference audio file
            output_dir: Directory to save output files
            emotion: Target emotion
            combine_output: Whether to combine chunks into a single file
            
        Returns:
            Path to final output file
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Split text into manageable chunks
        text_chunks = prepare_text_for_tts(text)
        print(f"Split text into {len(text_chunks)} chunks")
        
        # Extract speaker embedding once
        print(f"Extracting speaker embedding from {reference_audio}...")
        speaker_embedding = self.model.get_speaker_embedding(reference_audio)
        self.cached_speaker_embedding = speaker_embedding
        
        # Generate speech for each chunk
        output_files = []
        for i, chunk in enumerate(tqdm.tqdm(text_chunks, desc="Generating speech chunks")):
            # Skip empty chunks
            if not chunk.strip():
                continue
                
            # Generate output path for this chunk
            chunk_path = os.path.join(output_dir, f"chunk_{i+1:03d}.wav")
            
            # Generate speech
            self.generate_speech(
                text=chunk,
                reference_audio=None,  # Use cached embedding
                output_path=chunk_path,
                emotion=emotion,
                cache_speaker=True
            )
            
            output_files.append(chunk_path)
        
        # Combine chunks if requested
        if combine_output and output_files:
            import wave
            import contextlib
            
            combined_path = os.path.join(output_dir, "combined_output.wav")
            
            # Get parameters from first file
            with contextlib.closing(wave.open(output_files[0], 'rb')) as w:
                params = w.getparams()
            
            # Combine files
            with contextlib.closing(wave.open(combined_path, 'wb')) as output:
                output.setparams(params)
                
                for file in output_files:
                    with contextlib.closing(wave.open(file, 'rb')) as w:
                        output.writeframes(w.readframes(w.getnframes()))
            
            print(f"Combined output saved to {combined_path}")
            return combined_path
        
        return output_files[0] if output_files else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Cloning Inference")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--text", type=str, required=True,
                       help="Text to synthesize")
    parser.add_argument("--reference", type=str, required=True,
                       help="Path to reference voice audio")
    parser.add_argument("--output", type=str, default="output.wav",
                       help="Path to save output audio")
    parser.add_argument("--emotion", type=str, default="neutral",
                       choices=["neutral", "happy", "sad", "angry", "surprised", "fear"],
                       help="Emotion for synthesis")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cpu or cuda)")
    parser.add_argument("--no_emotion_effects", action="store_true",
                       help="Disable additional emotion effects")
    
    args = parser.parse_args()
    
    # Initialize inference module
    inference = VoiceCloningInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_emotion_effects=not args.no_emotion_effects
    )
    
    # Generate speech
    output_path = inference.generate_speech(
        text=args.text,
        reference_audio=args.reference,
        output_path=args.output,
        emotion=args.emotion
    )
    
    print("Speech generation complete!")