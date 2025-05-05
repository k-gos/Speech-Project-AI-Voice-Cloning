import os
import sys
import argparse
import torch
import torchaudio
import numpy as np
import pandas as pd
import json
import re
import yaml
import librosa
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging
import shutil
import soundfile

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """Class for preprocessing audio files for voice cloning"""
    
    def __init__(self, config):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Dictionary containing audio parameters
        """
        self.config = config
        self.audio_config = config["audio"]
        
        # Extract audio parameters
        self.sample_rate = self.audio_config["sampling_rate"]
        self.n_fft = self.audio_config["n_fft"]
        self.hop_length = self.audio_config["hop_length"]
        self.win_length = self.audio_config["win_length"]
        self.n_mels = self.audio_config["mel_channels"]
        self.fmin = self.audio_config["fmin"]
        self.fmax = self.audio_config["fmax"]
        self.max_audio_length = self.audio_config["max_audio_length"]
        
    def process_audio(self, file_path):
        """
        Process a single audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with processed audio data or None if failed
        """
        try:
            # Load audio file
            waveform, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Validate audio
            if len(waveform) < self.sample_rate * 0.5:  # Shorter than 0.5 seconds
                logger.warning(f"Audio too short: {file_path}")
                return None
                
            # Trim silence
            waveform, _ = librosa.effects.trim(waveform, top_db=20)
            
            # Normalize audio to -24 dB
            waveform = self.normalize_audio(waveform)
            
            # Ensure audio is not too long
            if len(waveform) > self.sample_rate * self.max_audio_length:
                logger.info(f"Trimming audio longer than {self.max_audio_length}s: {file_path}")
                waveform = waveform[:int(self.sample_rate * self.max_audio_length)]
            
            # Extract mel spectrogram
            mel_spectrogram = self.extract_mel_spectrogram(waveform)
            
            # Return processed data
            return {
                "file_path": str(file_path),
                "waveform": waveform,
                "mel_spectrogram": mel_spectrogram,
                "duration": len(waveform) / self.sample_rate,
                "sample_rate": self.sample_rate
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None
    
    def normalize_audio(self, waveform):
        """
        Normalize audio to target dB level
        
        Args:
            waveform: Audio waveform
            
        Returns:
            Normalized waveform
        """
        # Target dB level
        target_db = -24.0
        
        # Calculate RMS
        rms = np.sqrt(np.mean(waveform ** 2))
        
        if rms > 0:
            # Calculate gain
            target_rms = 10 ** (target_db / 20)
            gain = target_rms / rms
            
            # Apply gain
            return waveform * gain
        else:
            return waveform
    
    def extract_mel_spectrogram(self, waveform):
        """
        Extract mel spectrogram from waveform
        
        Args:
            waveform: Audio waveform
            
        Returns:
            Mel spectrogram
        """
        # Extract mel spectrogram using librosa
        mel_spectrogram = librosa.feature.melspectrogram(
            y=waveform, 
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to log scale
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        return mel_spectrogram
    
    def extract_pitch(self, waveform):
        """
        Extract pitch (F0) contour from waveform
        
        Args:
            waveform: Audio waveform
            
        Returns:
            Pitch contour
        """
        # Extract pitch using PYIN
        f0, voiced_flag, _ = librosa.pyin(
            waveform, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        return f0

class DatasetPreprocessor:
    """Class for preprocessing specific datasets for voice cloning"""
    
    def __init__(self, config, audio_processor, output_dir, cache_dir=None):
        """
        Initialize dataset preprocessor
        
        Args:
            config: Configuration dictionary
            audio_processor: AudioPreprocessor instance
            output_dir: Output directory for processed data
            cache_dir: Cache directory for intermediate results
        """
        self.config = config
        self.audio_processor = audio_processor
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directories for train, validation, and test sets
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.test_dir = self.output_dir / "test"
        
        self.train_dir.mkdir(exist_ok=True)
        self.val_dir.mkdir(exist_ok=True)
        self.test_dir.mkdir(exist_ok=True)
        
    def preprocess_ljspeech(self, input_dir):
        """
        Preprocess LJSpeech dataset
        
        Args:
            input_dir: Path to LJSpeech dataset directory
            
        Returns:
            Dictionary with dataset statistics
        """
        logger.info("Preprocessing LJSpeech dataset...")
        
        input_dir = Path(input_dir)
        logger.info(input_dir)
        metadata_path = input_dir / "metadata.csv"
        wavs_dir = input_dir / "wavs"
        
        # Check if metadata exists
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.csv not found in {input_dir}")
        
        # Read metadata
        metadata = pd.read_csv(
            metadata_path, 
            sep="|", 
            header=None, 
            names=["file_id", "transcription", "normalized_text"]
        )
        
        # Create speaker directory
        speaker_dir = {
            "train": self.train_dir / "LJ",
            "val": self.val_dir / "LJ",
            "test": self.test_dir / "LJ"
        }
        
        for dir_path in speaker_dir.values():
            dir_path.mkdir(exist_ok=True)
        
        # Split dataset
        n_samples = len(metadata)
        indices = np.random.permutation(n_samples)
        
        train_idx = indices[:int(0.8 * n_samples)]
        val_idx = indices[int(0.8 * n_samples):int(0.9 * n_samples)]
        test_idx = indices[int(0.9 * n_samples):]
        
        splits = {
            "train": {"indices": train_idx, "dir": speaker_dir["train"]},
            "val": {"indices": val_idx, "dir": speaker_dir["val"]},
            "test": {"indices": test_idx, "dir": speaker_dir["test"]}
        }
        
        # Process each split
        all_metadata = []
        
        for split_name, split_info in splits.items():
            logger.info(f"Processing {split_name} split...")
            
            split_metadata = []
            
            for idx in tqdm(split_info["indices"]):
                row = metadata.iloc[idx]
                file_id = row["file_id"]
                text = row["normalized_text"]
                
                wav_path = wavs_dir / f"{file_id}.wav"
                
                # Process audio
                result = self.audio_processor.process_audio(str(wav_path))
                
                if result:
                    # Save processed audio
                    output_path = split_info["dir"] / f"{file_id}.wav"
                    
                    # Save waveform
                    waveform = result["waveform"]
                    soundfile.write(str(output_path), waveform, result["sample_rate"])
                    
                    # Save mel spectrogram
                    mel_path = output_path.with_suffix(".npy")
                    np.save(str(mel_path), result["mel_spectrogram"])
                    
                    # Create metadata entry
                    entry = {
                        "file_path": str(output_path.relative_to(self.output_dir)),
                        "mel_path": str(mel_path.relative_to(self.output_dir)),
                        "text": text,
                        "speaker_id": "LJ",
                        "duration": result["duration"],
                        "split": split_name
                    }
                    
                    split_metadata.append(entry)
            
            # Save split metadata
            split_file = self.output_dir / f"{split_name}_metadata.json"
            with open(split_file, 'w') as f:
                json.dump(split_metadata, f, indent=2)
                
            all_metadata.extend(split_metadata)
            
            logger.info(f"Processed {len(split_metadata)} files for {split_name} split")
        
        # Save complete metadata
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        stats = {
            "dataset": "ljspeech",
            "train_samples": len(splits["train"]["indices"]),
            "val_samples": len(splits["val"]["indices"]),
            "test_samples": len(splits["test"]["indices"]),
            "total_samples": len(all_metadata)
        }
        
        return stats
    
    def preprocess_vctk(self, input_dir):
        """
        Preprocess VCTK dataset
        
        Args:
            input_dir: Path to VCTK dataset directory
            
        Returns:
            Dictionary with dataset statistics
        """
        logger.info("Preprocessing VCTK dataset...")
        
        input_dir = Path(input_dir)
        wav48_dir = input_dir / "wav48"
        txt_dir = input_dir / "txt"
        
        # Check if directories exist
        if not wav48_dir.exists():
            # Try alternative structure
            wav48_dir = input_dir / "VCTK-Corpus" / "wav48"
            txt_dir = input_dir / "VCTK-Corpus" / "txt"
            
            if not wav48_dir.exists():
                raise FileNotFoundError(f"wav48 directory not found in {input_dir}")
        
        # Get all speakers
        speakers = [d for d in wav48_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(speakers)} speakers")
        
        # Split speakers into train, validation, and test sets
        np.random.shuffle(speakers)
        
        n_speakers = len(speakers)
        train_speakers = speakers[:int(0.8 * n_speakers)]
        val_speakers = speakers[int(0.8 * n_speakers):int(0.9 * n_speakers)]
        test_speakers = speakers[int(0.9 * n_speakers):]
        
        splits = {
            "train": {"speakers": train_speakers, "dir": self.train_dir},
            "val": {"speakers": val_speakers, "dir": self.val_dir},
            "test": {"speakers": test_speakers, "dir": self.test_dir}
        }
        
        # Process each split
        all_metadata = []
        
        for split_name, split_info in splits.items():
            logger.info(f"Processing {split_name} split ({len(split_info['speakers'])} speakers)...")
            
            split_metadata = []
            
            for speaker_dir in tqdm(split_info["speakers"]):
                speaker_id = speaker_dir.name
                
                # Create speaker directory
                output_speaker_dir = split_info["dir"] / speaker_id
                output_speaker_dir.mkdir(exist_ok=True)
                
                # Get all wav files for speaker
                wav_files = list(speaker_dir.glob("*.wav"))
                
                # Limit number of samples per speaker to prevent imbalance
                max_samples_per_speaker = 100 if split_name == "train" else 20
                if len(wav_files) > max_samples_per_speaker:
                    np.random.shuffle(wav_files)
                    wav_files = wav_files[:max_samples_per_speaker]
                
                for wav_path in wav_files:
                    file_id = wav_path.stem
                    
                    # Find corresponding text file
                    txt_path = txt_dir / speaker_id / f"{file_id}.txt"
                    
                    if txt_path.exists():
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                    else:
                        # Skip files without text
                        logger.warning(f"Text file not found for {wav_path}")
                        continue
                    
                    # Process audio
                    result = self.audio_processor.process_audio(str(wav_path))
                    
                    if result:
                        # Save processed audio
                        output_path = output_speaker_dir / f"{file_id}.wav"
                        
                        # Save waveform
                        waveform = result["waveform"]
                        soundfile.write(str(output_path), waveform, result["sample_rate"])
                        
                        # Save mel spectrogram
                        mel_path = output_path.with_suffix(".npy")
                        np.save(str(mel_path), result["mel_spectrogram"])
                        
                        # Create metadata entry
                        entry = {
                            "file_path": str(output_path.relative_to(self.output_dir)),
                            "mel_path": str(mel_path.relative_to(self.output_dir)),
                            "text": text,
                            "speaker_id": speaker_id,
                            "duration": result["duration"],
                            "split": split_name
                        }
                        
                        split_metadata.append(entry)
            
            # Save split metadata
            split_file = self.output_dir / f"{split_name}_metadata.json"
            with open(split_file, 'w') as f:
                json.dump(split_metadata, f, indent=2)
                
            all_metadata.extend(split_metadata)
            
            logger.info(f"Processed {len(split_metadata)} files for {split_name} split")
        
        # Save complete metadata
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        stats = {
            "dataset": "vctk",
            "train_speakers": len(splits["train"]["speakers"]),
            "val_speakers": len(splits["val"]["speakers"]),
            "test_speakers": len(splits["test"]["speakers"]),
            "total_samples": len(all_metadata)
        }
        
        return stats
    
    def preprocess_libri_tts(self, input_dir):
        """
        Preprocess LibriTTS dataset
        
        Args:
            input_dir: Path to LibriTTS dataset directory
            
        Returns:
            Dictionary with dataset statistics
        """
        logger.info("Preprocessing LibriTTS dataset...")
        
        input_dir = Path(input_dir)
        
        # LibriTTS has a different structure, find all transcript files recursively
        transcript_files = list(input_dir.glob("*/*/*.trans.tsv"))
        
        if not transcript_files:
            raise FileNotFoundError(f"No transcript files found in {input_dir}")
        
        # Group speakers by their first digit to ensure even distribution
        speaker_groups = {}
        
        for trans_file in transcript_files:
            # Extract speaker ID from path
            speaker_id = trans_file.parent.name
            group = speaker_id[0]  # First digit
            
            if group not in speaker_groups:
                speaker_groups[group] = []
                
            speaker_groups[group].append((speaker_id, trans_file))
        
        # Assign speakers to splits
        train_speakers = []
        val_speakers = []
        test_speakers = []
        
        for group, speakers in speaker_groups.items():
            np.random.shuffle(speakers)
            n_speakers = len(speakers)
            
            train_speakers.extend(speakers[:int(0.8 * n_speakers)])
            val_speakers.extend(speakers[int(0.8 * n_speakers):int(0.9 * n_speakers)])
            test_speakers.extend(speakers[int(0.9 * n_speakers):])
        
        splits = {
            "train": {"speakers": train_speakers, "dir": self.train_dir},
            "val": {"speakers": val_speakers, "dir": self.val_dir},
            "test": {"speakers": test_speakers, "dir": self.test_dir}
        }
        
        # Process each split
        all_metadata = []
        
        for split_name, split_info in splits.items():
            logger.info(f"Processing {split_name} split ({len(split_info['speakers'])} speakers)...")
            
            split_metadata = []
            
            for speaker_id, trans_file in tqdm(split_info["speakers"]):
                # Create speaker directory
                output_speaker_dir = split_info["dir"] / speaker_id
                output_speaker_dir.mkdir(exist_ok=True)
                
                # Read transcript file
                with open(trans_file, 'r', encoding='utf-8') as f:
                    transcript_lines = f.readlines()
                
                # Process each line
                for line in transcript_lines:
                    parts = line.strip().split("\t")
                    if len(parts) != 2:
                        continue
                        
                    file_id, text = parts
                    
                    # Create wav file path
                    wav_path = trans_file.parent / f"{file_id}.wav"
                    
                    if not wav_path.exists():
                        # Skip files without audio
                        logger.warning(f"Audio file not found: {wav_path}")
                        continue
                    
                    # Process audio
                    result = self.audio_processor.process_audio(str(wav_path))
                    
                    if result:
                        # Save processed audio
                        output_path = output_speaker_dir / f"{file_id}.wav"
                        
                        # Save waveform
                        waveform = result["waveform"]
                        soundfile.write(str(output_path), waveform, result["sample_rate"])
                        
                        # Save mel spectrogram
                        mel_path = output_path.with_suffix(".npy")
                        np.save(str(mel_path), result["mel_spectrogram"])
                        
                        # Create metadata entry
                        entry = {
                            "file_path": str(output_path.relative_to(self.output_dir)),
                            "mel_path": str(mel_path.relative_to(self.output_dir)),
                            "text": text,
                            "speaker_id": speaker_id,
                            "duration": result["duration"],
                            "split": split_name
                        }
                        
                        split_metadata.append(entry)
            
            # Save split metadata
            split_file = self.output_dir / f"{split_name}_metadata.json"
            with open(split_file, 'w') as f:
                json.dump(split_metadata, f, indent=2)
                
            all_metadata.extend(split_metadata)
            
            logger.info(f"Processed {len(split_metadata)} files for {split_name} split")
        
        # Save complete metadata
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        stats = {
            "dataset": "libri_tts",
            "train_speakers": len({s[0] for s in splits["train"]["speakers"]}),
            "val_speakers": len({s[0] for s in splits["val"]["speakers"]}),
            "test_speakers": len({s[0] for s in splits["test"]["speakers"]}),
            "total_samples": len(all_metadata)
        }
        
        return stats
    
    def preprocess_common_voice(self, input_dir):
        """
        Preprocess Common Voice dataset
        
        Args:
            input_dir: Path to Common Voice dataset directory
            
        Returns:
            Dictionary with dataset statistics
        """
        logger.info("Preprocessing Common Voice dataset...")
        
        input_dir = Path(input_dir)
        clips_dir = input_dir / "clips"
        
        # Common Voice has CSV files for different splits
        train_csv = input_dir / "train.tsv"
        dev_csv = input_dir / "dev.tsv"
        test_csv = input_dir / "test.tsv"
        
        if not (train_csv.exists() and dev_csv.exists() and test_csv.exists()):
            raise FileNotFoundError(f"TSV files not found in {input_dir}")
        
        # Read CSV files
        train_df = pd.read_csv(train_csv, sep="\t")
        dev_df = pd.read_csv(dev_csv, sep="\t")
        test_df = pd.read_csv(test_csv, sep="\t")
        
        splits = {
            "train": {"df": train_df, "dir": self.train_dir},
            "val": {"df": dev_df, "dir": self.val_dir},
            "test": {"df": test_df, "dir": self.test_dir}
        }
        
        # Process each split
        all_metadata = []
        
        for split_name, split_info in splits.items():
            logger.info(f"Processing {split_name} split ({len(split_info['df'])} samples)...")
            
            split_metadata = []
            
            # Limit number of samples for each split to prevent excessive processing time
            max_samples = 10000 if split_name == "train" else 2000
            df = split_info["df"].sample(min(max_samples, len(split_info["df"]))).copy()
            
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                # Extract information
                file_name = row["path"]
                text = row["sentence"]
                client_id = row.get("client_id", str(idx))  # Use index if client_id not available
                
                # Create wav file path
                wav_path = clips_dir / file_name
                
                if not wav_path.exists():
                    # Skip files without audio
                    logger.warning(f"Audio file not found: {wav_path}")
                    continue
                
                # Process audio
                result = self.audio_processor.process_audio(str(wav_path))
                
                if result:
                    # Create speaker directory (using client_id as speaker)
                    speaker_id = client_id[:10]  # Trim to reasonable length
                    output_speaker_dir = split_info["dir"] / speaker_id
                    output_speaker_dir.mkdir(exist_ok=True)
                    
                    # Save processed audio
                    output_file_name = f"{Path(file_name).stem}.wav"
                    output_path = output_speaker_dir / output_file_name
                    
                    # Save waveform
                    waveform = result["waveform"]
                    soundfile.write(str(output_path), waveform, result["sample_rate"])
                    
                    # Save mel spectrogram
                    mel_path = output_path.with_suffix(".npy")
                    np.save(str(mel_path), result["mel_spectrogram"])
                    
                    # Create metadata entry
                    entry = {
                        "file_path": str(output_path.relative_to(self.output_dir)),
                        "mel_path": str(mel_path.relative_to(self.output_dir)),
                        "text": text,
                        "speaker_id": speaker_id,
                        "duration": result["duration"],
                        "split": split_name
                    }
                    
                    split_metadata.append(entry)
            
            # Save split metadata
            split_file = self.output_dir / f"{split_name}_metadata.json"
            with open(split_file, 'w') as f:
                json.dump(split_metadata, f, indent=2)
                
            all_metadata.extend(split_metadata)
            
            logger.info(f"Processed {len(split_metadata)} files for {split_name} split")
        
        # Save complete metadata
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        stats = {
            "dataset": "common_voice",
            "train_samples": len([m for m in all_metadata if m["split"] == "train"]),
            "val_samples": len([m for m in all_metadata if m["split"] == "val"]),
            "test_samples": len([m for m in all_metadata if m["split"] == "test"]),
            "total_samples": len(all_metadata),
            "unique_speakers": len(set(m["speaker_id"] for m in all_metadata))
        }
        
        return stats
    
    def preprocess_aishell3(self, input_dir):
        """
        Preprocess AISHELL-3 dataset
        
        Args:
            input_dir: Path to AISHELL-3 dataset directory
            
        Returns:
            Dictionary with dataset statistics
        """
        logger.info("Preprocessing AISHELL-3 dataset...")
        
        input_dir = Path(input_dir)
        wav_dir = input_dir / "wav"
        
        # AISHELL-3 has train/test directories with speakers
        train_dir = wav_dir / "train"
        test_dir = wav_dir / "test"
        
        if not (train_dir.exists() and test_dir.exists()):
            raise FileNotFoundError(f"Train/test directories not found in {input_dir}")
        
        # Read transcript
        transcript_path = input_dir / "train" / "label_train-set.txt"
        test_transcript_path = input_dir / "test" / "label_test-set.txt"
        
        # Process training speakers
        train_speakers = list(train_dir.iterdir())
        np.random.shuffle(train_speakers)
        
        n_train_speakers = len(train_speakers)
        val_speakers = train_speakers[int(0.8 * n_train_speakers):]
        train_speakers = train_speakers[:int(0.8 * n_train_speakers)]
        
        # Test speakers
        test_speakers = list(test_dir.iterdir())
        
        splits = {
            "train": {"speakers": train_speakers, "dir": self.train_dir, "transcript": transcript_path},
            "val": {"speakers": val_speakers, "dir": self.val_dir, "transcript": transcript_path},
            "test": {"speakers": test_speakers, "dir": self.test_dir, "transcript": test_transcript_path}
        }
        
        # Read transcripts
        def read_transcript(path):
            transcript = {}
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('|')
                        if len(parts) >= 2:
                            file_id = parts[0]
                            text = parts[1]
                            transcript[file_id] = text
            except Exception as e:
                logger.error(f"Error reading transcript file {path}: {str(e)}")
            return transcript
        
        train_transcript = read_transcript(transcript_path)
        test_transcript = read_transcript(test_transcript_path)
        
        # Process each split
        all_metadata = []
        
        for split_name, split_info in splits.items():
            logger.info(f"Processing {split_name} split ({len(split_info['speakers'])} speakers)...")
            
            split_metadata = []
            transcript = train_transcript if "train" in split_name or "val" in split_name else test_transcript
            
            for speaker_dir in tqdm(split_info["speakers"]):
                speaker_id = speaker_dir.name
                
                # Create speaker directory
                output_speaker_dir = split_info["dir"] / speaker_id
                output_speaker_dir.mkdir(exist_ok=True)
                
                # Get all wav files for speaker
                wav_files = list(speaker_dir.glob("*.wav"))
                
                # Limit number of samples per speaker to prevent imbalance
                max_samples_per_speaker = 100 if split_name == "train" else 20
                if len(wav_files) > max_samples_per_speaker:
                    np.random.shuffle(wav_files)
                    wav_files = wav_files[:max_samples_per_speaker]
                
                for wav_path in wav_files:
                    file_id = wav_path.stem
                    
                    # Get text from transcript
                    text = transcript.get(file_id, "")
                    
                    if not text:
                        # Skip files without text
                        logger.warning(f"No transcript found for {wav_path}")
                        continue
                    
                    # Process audio
                    result = self.audio_processor.process_audio(str(wav_path))
                    
                    if result:
                        # Save processed audio
                        output_path = output_speaker_dir / f"{file_id}.wav"
                        
                        # Save waveform
                        waveform = result["waveform"]
                        soundfile.write(str(output_path), waveform, result["sample_rate"])
                        
                        # Save mel spectrogram
                        mel_path = output_path.with_suffix(".npy")
                        np.save(str(mel_path), result["mel_spectrogram"])
                        
                        # Create metadata entry
                        entry = {
                            "file_path": str(output_path.relative_to(self.output_dir)),
                            "mel_path": str(mel_path.relative_to(self.output_dir)),
                            "text": text,
                            "speaker_id": speaker_id,
                            "duration": result["duration"],
                            "split": split_name
                        }
                        
                        split_metadata.append(entry)
            
            # Save split metadata
            split_file = self.output_dir / f"{split_name}_metadata.json"
            with open(split_file, 'w') as f:
                json.dump(split_metadata, f, indent=2)
                
            all_metadata.extend(split_metadata)
            
            logger.info(f"Processed {len(split_metadata)} files for {split_name} split")
        
        # Save complete metadata
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        stats = {
            "dataset": "aishell3",
            "train_speakers": len(splits["train"]["speakers"]),
            "val_speakers": len(splits["val"]["speakers"]),
            "test_speakers": len(splits["test"]["speakers"]),
            "total_samples": len(all_metadata)
        }
        
        return stats
    
    def preprocess(self, dataset_name, input_dir):
        """
        Preprocess dataset based on name
        
        Args:
            dataset_name: Name of dataset
            input_dir: Path to dataset directory
            
        Returns:
            Dictionary with dataset statistics
        """
        if dataset_name == "ljspeech":
            return self.preprocess_ljspeech(input_dir)
        elif dataset_name == "vctk":
            return self.preprocess_vctk(input_dir)
        elif dataset_name == "libri_tts":
            return self.preprocess_libri_tts(input_dir)
        elif dataset_name == "common_voice":
            return self.preprocess_common_voice(input_dir)
        elif dataset_name == "aishell3":
            return self.preprocess_aishell3(input_dir)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for voice cloning")
    
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (ljspeech, vctk, libri_tts, common_voice, aishell3)")
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Path to input dataset directory")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Path to output directory")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config file")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of worker processes")
    parser.add_argument("--max-audio-len", type=float, default=10.0,
                       help="Maximum audio length in seconds")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Directory for caching intermediate results")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override max audio length if specified
    if args.max_audio_len:
        config["audio"]["max_audio_length"] = args.max_audio_len
    
    # Initialize audio processor and dataset preprocessor
    try:
        import soundfile  # Import here to catch missing dependency
    except ImportError:
        logger.error("soundfile package not found. Please install it with: pip install soundfile")
        sys.exit(1)
    
    audio_processor = AudioPreprocessor(config)
    dataset_preprocessor = DatasetPreprocessor(
        config, 
        audio_processor, 
        args.output_dir, 
        args.cache_dir
    )
    
    # Set number of workers for parallel processing
    if args.num_workers > 0:
        torch.set_num_threads(args.num_workers)
    
    # Preprocess dataset
    try:
        stats = dataset_preprocessor.preprocess(args.dataset, args.input_dir)
        
        # Save stats
        with open(os.path.join(args.output_dir, "stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Dataset preprocessing complete. Stats: {stats}")
        
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {str(e)}")
        raise
    
if __name__ == "__main__":
    main()