"""
Dataset Loader Module

Provides dataset loading and preprocessing for various speech datasets
used for voice cloning.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import librosa
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import random

class VoiceCloningDataset(Dataset):
    """Dataset for voice cloning training"""
    
    def __init__(self, 
                 root_path: str, 
                 dataset_name: str,
                 split: str = "train",
                 max_audio_length: int = 10,  # in seconds
                 sampling_rate: int = 22050,
                 mel_channels: int = 80,
                 n_fft: int = 1024,
                 win_length: int = 1024,
                 hop_length: int = 256,
                 cache_dir: Optional[str] = None):
        """
        Args:
            root_path: Path to dataset directory
            dataset_name: Name of dataset ('libri_tts', 'vctk', 'common_voice', 'aishell3')
            split: Data split ('train', 'dev', 'test')
            max_audio_length: Maximum audio length in seconds
            sampling_rate: Target sampling rate
            mel_channels: Number of mel spectrogram channels
            n_fft: FFT size
            win_length: Window length for STFT
            hop_length: Hop length for STFT
            cache_dir: Directory to cache processed features (optional)
        """
        self.root_path = Path(root_path)
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.max_audio_length = max_audio_length
        self.max_samples = max_audio_length * sampling_rate
        
        # Audio processing parameters
        self.sampling_rate = sampling_rate
        self.mel_channels = mel_channels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        
        # Setup cache directory if provided
        self.cache_dir = None
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Load appropriate dataset
        if self.dataset_name == 'libri_tts':
            self.metadata = self._load_libri_tts()
        elif self.dataset_name == 'vctk':
            self.metadata = self._load_vctk()
        elif self.dataset_name == 'common_voice':
            self.metadata = self._load_common_voice()
        elif self.dataset_name == 'aishell3':
            self.metadata = self._load_aishell3()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        # Create speaker ID mapping
        self.speaker_ids = sorted(list(set([item['speaker_id'] for item in self.metadata])))
        self.speaker_id_map = {speaker: idx for idx, speaker in enumerate(self.speaker_ids)}
        
        print(f"Loaded {len(self.metadata)} samples from {self.dataset_name} ({self.split}) with {len(self.speaker_ids)} speakers")
        
    def _load_libri_tts(self) -> List[Dict]:
        """Load LibriTTS dataset metadata"""
        metadata = []
        
        # Select appropriate split directories
        if self.split == "train":
            split_dirs = ["train-clean-100", "train-clean-360"]
        elif self.split == "dev":
            split_dirs = ["dev-clean"]
        else:  # test
            split_dirs = ["test-clean"]
            
        for split_dir in split_dirs:
            split_path = self.root_path / split_dir
            
            if not split_path.exists():
                print(f"Warning: {split_path} does not exist")
                continue
                
            # LibriTTS structure: {split}/{speaker_id}/{book_id}/{speaker_id}_{book_id}_{utterance_id}.wav
            for speaker_id in os.listdir(split_path):
                speaker_path = split_path / speaker_id
                if not speaker_path.is_dir():
                    continue
                    
                for book_id in os.listdir(speaker_path):
                    book_path = speaker_path / book_id
                    if not book_path.is_dir():
                        continue
                        
                    for file in book_path.glob("*.wav"):
                        # Get corresponding text file
                        text_file = book_path / f"{file.stem}.normalized.txt"
                        if text_file.exists():
                            with open(text_file, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                                
                            metadata.append({
                                "speaker_id": speaker_id,
                                "audio_path": str(file),
                                "text": text
                            })
                            
        return metadata
    
    def _load_vctk(self) -> List[Dict]:
        """Load VCTK dataset metadata"""
        metadata = []
        
        # VCTK structure: wav48_silence_trimmed/{speaker_id}/{utterance_id}.wav
        wav_dir = self.root_path / "wav48_silence_trimmed"
        txt_dir = self.root_path / "txt"
        
        if not wav_dir.exists():
            print(f"Warning: {wav_dir} does not exist")
            return metadata
            
        # Split speakers for train/dev/test (90/5/5 split)
        all_speakers = sorted([d for d in os.listdir(wav_dir) if d.startswith('p')])
        n_speakers = len(all_speakers)
        
        if self.split == "train":
            speakers = all_speakers[:int(0.9 * n_speakers)]
        elif self.split == "dev":
            speakers = all_speakers[int(0.9 * n_speakers):int(0.95 * n_speakers)]
        else:  # test
            speakers = all_speakers[int(0.95 * n_speakers):]
        
        for speaker_id in speakers:
            speaker_wav_path = wav_dir / speaker_id
            speaker_txt_path = txt_dir / speaker_id
            
            if not speaker_wav_path.is_dir():
                continue
                
            for wav_file in speaker_wav_path.glob("*.wav"):
                # Get corresponding text file
                txt_file = speaker_txt_path / f"{wav_file.stem}.txt"
                
                if txt_file.exists():
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        
                    metadata.append({
                        "speaker_id": speaker_id,
                        "audio_path": str(wav_file),
                        "text": text
                    })
                    
        return metadata
    
    def _load_common_voice(self) -> List[Dict]:
        """Load Common Voice dataset metadata"""
        metadata = []
        
        # Common Voice structure: {lang}/{split}.tsv + /clips/ folder with audio
        tsv_file = self.root_path / f"{self.split}.tsv"
        clips_dir = self.root_path / "clips"
        
        if not tsv_file.exists() or not clips_dir.exists():
            print(f"Warning: {tsv_file} or {clips_dir} does not exist")
            return metadata
            
        # Read TSV file
        df = pd.read_csv(tsv_file, sep='\t')
        
        # Process entries
        for _, row in df.iterrows():
            audio_path = clips_dir / row['path']
            
            if audio_path.exists():
                metadata.append({
                    "speaker_id": row['client_id'],
                    "audio_path": str(audio_path),
                    "text": row['sentence']
                })
                
        return metadata
    
    def _load_aishell3(self) -> List[Dict]:
        """Load AISHELL-3 dataset metadata"""
        metadata = []
        
        # AISHELL-3 structure: {split}/wav/{speaker_id}/{utterance_id}.wav
        wav_dir = self.root_path / self.split / "wav"
        transcript_file = self.root_path / self.split / "content.txt"
        
        if not wav_dir.exists() or not transcript_file.exists():
            print(f"Warning: {wav_dir} or {transcript_file} does not exist")
            return metadata
            
        # Load transcripts
        transcripts = {}
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    utterance_id = parts[0]
                    text = ' '.join(parts[1:])
                    transcripts[utterance_id] = text
        
        # Find audio files
        for speaker_id in os.listdir(wav_dir):
            speaker_path = wav_dir / speaker_id
            
            if not speaker_path.is_dir():
                continue
                
            for audio_file in speaker_path.glob("*.wav"):
                utterance_id = audio_file.stem
                
                if utterance_id in transcripts:
                    metadata.append({
                        "speaker_id": speaker_id,
                        "audio_path": str(audio_file),
                        "text": transcripts[utterance_id]
                    })
                    
        return metadata
    
    def _get_cache_path(self, audio_path: str) -> Path:
        """Get cache file path for an audio file"""
        if self.cache_dir is None:
            return None
            
        # Create a unique filename based on the audio path
        audio_hash = str(hash(audio_path))
        return self.cache_dir / f"{audio_hash}.pt"
    
    def _load_or_process_audio(self, audio_path: str) -> Dict:
        """Load audio from cache or process it from scratch"""
        cache_path = self._get_cache_path(audio_path)
        
        # Try to load from cache
        if cache_path is not None and cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception as e:
                print(f"Error loading cache file {cache_path}: {str(e)}")
        
        # Process audio from scratch
        try:
            # Load and resample
            waveform, sr = librosa.load(audio_path, sr=self.sampling_rate)
            
            # Trim silence
            waveform, _ = librosa.effects.trim(waveform, top_db=20)
            
            # Ensure consistent length
            if len(waveform) > self.max_samples:
                waveform = waveform[:self.max_samples]
            
            # Compute mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=waveform, 
                sr=self.sampling_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.mel_channels
            )
            
            # Convert to log scale
            mel = librosa.power_to_db(mel, ref=np.max)
            
            # Normalize
            mel = (mel - mel.min()) / (mel.max() - mel.min()) * 2 - 1
            
            # Convert to tensors
            waveform_tensor = torch.FloatTensor(waveform)
            mel_tensor = torch.FloatTensor(mel)
            
            result = {
                "waveform": waveform_tensor,
                "mel_spectrogram": mel_tensor,
                "duration": len(waveform) / self.sampling_rate
            }
            
            # Save to cache
            if cache_path is not None:
                torch.save(result, cache_path)
                
            return result
            
        except Exception as e:
            print(f"Error processing audio {audio_path}: {str(e)}")
            # Return empty tensors as fallback
            return {
                "waveform": torch.zeros(self.sampling_rate),
                "mel_spectrogram": torch.zeros(self.mel_channels, self.hop_length),
                "duration": 1.0
            }
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, index: int) -> Dict:
        item = self.metadata[index]
        
        # Load audio
        audio_path = item['audio_path']
        audio_data = self._load_or_process_audio(audio_path)
        
        # Get speaker ID as integer
        speaker_idx = self.speaker_id_map[item['speaker_id']]
        
        return {
            "waveform": audio_data["waveform"],
            "mel_spectrogram": audio_data["mel_spectrogram"],
            "text": item['text'],
            "speaker_id": speaker_idx,
            "speaker_name": item['speaker_id'],
            "duration": audio_data["duration"],
            "audio_path": item['audio_path']
        }


def collate_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched data with padded sequences
    """
    # Extract data
    waveforms = [item["waveform"] for item in batch]
    mels = [item["mel_spectrogram"] for item in batch]
    texts = [item["text"] for item in batch]
    speaker_ids = torch.tensor([item["speaker_id"] for item in batch], dtype=torch.long)
    speaker_names = [item["speaker_name"] for item in batch]
    durations = torch.tensor([item["duration"] for item in batch], dtype=torch.float)
    audio_paths = [item["audio_path"] for item in batch]
    
    # Get max lengths
    max_waveform_len = max([w.shape[0] for w in waveforms])
    max_mel_len = max([m.shape[1] for m in mels])
    
    # Pad waveforms
    waveform_lens = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    padded_waveforms = torch.zeros(len(batch), max_waveform_len)
    
    for i, waveform in enumerate(waveforms):
        padded_waveforms[i, :waveform.shape[0]] = waveform
    
    # Pad mel spectrograms
    mel_lens = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
    padded_mels = torch.zeros(len(batch), batch[0]["mel_spectrogram"].shape[0], max_mel_len)
    
    for i, mel in enumerate(mels):
        padded_mels[i, :, :mel.shape[1]] = mel
    
    # Create mel masks
    mel_masks = torch.zeros(len(batch), max_mel_len)
    for i, mel_len in enumerate(mel_lens):
        mel_masks[i, :mel_len] = 1
    
    return {
        "waveforms": padded_waveforms,
        "waveform_lens": waveform_lens,
        "mel_spectrograms": padded_mels,
        "mel_lens": mel_lens,
        "mel_masks": mel_masks,
        "texts": texts,
        "speaker_ids": speaker_ids,
        "speaker_names": speaker_names,
        "durations": durations,
        "audio_paths": audio_paths
    }


def get_dataloader(root_path: str, 
                   dataset_name: str,
                   split: str = "train",
                   batch_size: int = 16,
                   shuffle: bool = None,
                   num_workers: int = 4,
                   cache_dir: Optional[str] = None) -> DataLoader:
    """
    Get DataLoader for the specified dataset
    
    Args:
        root_path: Path to dataset directory
        dataset_name: Name of dataset
        split: Data split
        batch_size: Batch size
        shuffle: Whether to shuffle data (default: True for train, False otherwise)
        num_workers: Number of worker processes
        cache_dir: Directory to cache processed features
        
    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = (split == "train")
    
    dataset = VoiceCloningDataset(
        root_path=root_path, 
        dataset_name=dataset_name, 
        split=split,
        cache_dir=cache_dir
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test the dataset loader
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description="Test dataset loader")
    parser.add_argument("--dataset", type=str, default="vctk", 
                       choices=["libri_tts", "vctk", "common_voice", "aishell3"],
                       help="Dataset name")
    parser.add_argument("--path", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Path to cache directory")
    
    args = parser.parse_args()
    
    # Create dataloader
    dataloader = get_dataloader(
        root_path=args.path,
        dataset_name=args.dataset,
        split="train",
        batch_size=2,
        cache_dir=args.cache_dir
    )
    
    # Get a sample batch
    batch = next(iter(dataloader))
    
    print(f"Batch contents:")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape}, {value.dtype}")
        elif isinstance(value, list):
            print(f"  {key}: List[{len(value)}]")
            if value:
                print(f"    First item: {value[0]}")
    
    # Visualize mel spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(batch["mel_spectrograms"][0].numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(f"Mel Spectrogram - Speaker: {batch['speaker_names'][0]}")
    plt.tight_layout()
    plt.savefig("sample_mel.png")
    print("Saved sample mel spectrogram to sample_mel.png")