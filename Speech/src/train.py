"""
Training script for voice cloning system
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from models.model import VoiceCloningModel, VoiceCloningLoss
from data.dataset_loader import get_dataloader
from utils.audio import save_audio


class Trainer:
    """Trainer for voice cloning model"""
    
    def __init__(self, args):
        """
        Initialize trainer
        
        Args:
            args: Command line arguments
        """
        self.args = args
        
        # Setup paths
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))
        
        # Load config
        self.config = self._load_config(args.config)
        
        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Initialize model
        self._init_model()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.0001),
            weight_decay=self.config.get('weight_decay', 0.0001)
        )
        
        # Initialize loss function
        self.criterion = VoiceCloningLoss(
            mel_loss_weight=self.config.get('mel_loss_weight', 1.0),
            feature_loss_weight=self.config.get('feature_loss_weight', 0.1)
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Initialize dataloaders
        self._init_dataloaders()
        
        # Tracking variables
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Load checkpoint if provided
        if args.checkpoint:
            self._load_checkpoint(args.checkpoint)
    
    def _load_config(self, config_path):
        """Load configuration file"""
        if os.path.exists(config_path):
            # Determine file type from extension
            if config_path.endswith('.json'):
                with open(config_path, 'r') as f:
                    return json.load(f)
            elif config_path.endswith(('.yaml', '.yml')):
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    def _init_model(self):
        """Initialize the model"""
        print("Initializing model...")

        # Get model configurations from config file
        model_config = self.config.get('model', {})

        # CRITICAL FIX: Create a clean config dict without unexpected arguments
        clean_config = {
            'd_model': model_config.get('d_model', 512),
            'use_emotion_encoder': model_config.get('use_emotion_encoder', False)
        }

        # Initialize model with proper parameters
        self.model = VoiceCloningModel(config=clean_config)

        # If speaker encoder path is provided, try to load it separately
        if hasattr(self.args, 'speaker_encoder') and self.args.speaker_encoder:
            print(f"Loading pretrained speaker encoder from {self.args.speaker_encoder}")
            try:
                # Check if load_pretrained method exists
                if hasattr(self.model.speaker_encoder, 'load_pretrained'):
                    self.model.speaker_encoder.load_pretrained(self.args.speaker_encoder)
                else:
                    print("Warning: speaker_encoder has no load_pretrained method")
            except Exception as e:
                print(f"Warning: Failed to load pretrained speaker encoder: {e}")

        # If vocoder path is provided, try to load it separately
        if hasattr(self.args, 'vocoder') and self.args.vocoder:
            print(f"Loading pretrained vocoder from {self.args.vocoder}")
            try:
                # Check if load_pretrained method exists
                if hasattr(self.model.vocoder, 'load_pretrained'):
                    self.model.vocoder.load_pretrained(self.args.vocoder)
                else:
                    print("Warning: vocoder has no load_pretrained method")
            except Exception as e:
                print(f"Warning: Failed to load pretrained vocoder: {e}")

        # Move model to device
        self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")
    
    def _init_dataloaders(self):
        """Initialize data loaders"""
        print("Initializing data loaders...")
        
        # Training dataloader
        self.train_loader = get_dataloader(
            root_path=self.args.data_dir,
            dataset_name=self.args.dataset,
            split="train",
            batch_size=self.config.get('batch_size', 16),
            num_workers=self.args.num_workers,
            max_audio_length=self.config.get('max_audio_length', 10),
            use_cache=True
        )
        
        # Validation dataloader
        self.val_loader = get_dataloader(
            root_path=self.args.data_dir,
            dataset_name=self.args.dataset,
            split="dev",
            batch_size=self.config.get('batch_size', 16),
            num_workers=self.args.num_workers,
            max_audio_length=self.config.get('max_audio_length', 10),
            use_cache=True
        )
        
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
    
    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load training state
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
            
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
            
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
            
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
            
        print(f"Resuming from epoch {self.start_epoch}")
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint (for resuming)
        latest_path = self.output_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if this is the best model
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
            
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def _log_metrics(self, metrics, step, prefix='train'):
        """Log metrics to tensorboard"""
        for key, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{key}', value, step)
    
    def _log_audio_samples(self, waveforms, sample_rate, step, prefix='train'):
        """Log audio samples to tensorboard"""
        # Log up to 3 samples
        for i, waveform in enumerate(waveforms[:3]):
            self.writer.add_audio(
                f'{prefix}/audio_sample_{i}',
                waveform.detach().cpu().numpy(),
                step,
                sample_rate=sample_rate
            )
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{self.args.epochs}") as pbar:
            for batch_idx, batch in enumerate(self.train_loader):
                # Move data to device
                texts = batch['texts']
                mel_specs = batch['mel_spectrograms'].to(self.device)
                waveforms = batch['waveforms'].to(self.device)
                speaker_ids = batch['speaker_ids'].to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model.forward(
                    text=texts,
                    target_mel=mel_specs,
                    speaker_embedding=None,  # Will be extracted from mel_specs
                    emotion="neutral"  # Default for training
                )
                
                # Calculate loss
                loss, loss_components = self.criterion(
                    predicted_mel=outputs['mel_output'],
                    target_mel=mel_specs
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('grad_clip_thresh', 1.0)
                )
                
                # Update parameters
                self.optimizer.step()
                
                # Update tracking variables
                batch_loss = loss.item()
                epoch_loss += batch_loss
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix(loss=f"{batch_loss:.4f}")
                pbar.update()
                
                # Log metrics periodically
                if batch_idx % self.args.log_interval == 0:
                    self._log_metrics(
                        {'loss': batch_loss, **{k: v.item() for k, v in loss_components.items()}},
                        self.global_step
                    )
                    
                    # Log audio samples
                    if batch_idx % (self.args.log_interval * 10) == 0:
                        with torch.no_grad():
                            # Generate speech for logging
                            gen_outputs = self.model.forward(
                                text=texts[0],  # Just use the first text in batch
                                speaker_embedding=outputs['speaker_embedding'][0:1],
                                emotion="neutral"
                            )
                            
                            if gen_outputs['waveform'] is not None:
                                self._log_audio_samples(
                                    [gen_outputs['waveform'][0], waveforms[0]],
                                    self.config.get('sampling_rate', 22050),
                                    self.global_step
                                )
                
                # Free up memory
                del outputs, loss, loss_components
                
        # Calculate average epoch loss
        epoch_loss /= num_batches
        self.train_losses.append(epoch_loss)
        
        return epoch_loss
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        val_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            with tqdm(total=num_batches, desc=f"Validation") as pbar:
                for batch in self.val_loader:
                    # Move data to device
                    texts = batch['texts']
                    mel_specs = batch['mel_spectrograms'].to(self.device)
                    speaker_ids = batch['speaker_ids'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model.forward(
                        text=texts,
                        target_mel=mel_specs,
                        speaker_embedding=None,  # Will be extracted from mel_specs
                        emotion="neutral"
                    )
                    
                    # Calculate loss
                    loss, _ = self.criterion(
                        predicted_mel=outputs['mel_output'],
                        target_mel=mel_specs
                    )
                    
                    # Update tracking
                    val_loss += loss.item()
                    pbar.update()
                    
        # Calculate average validation loss
        val_loss /= num_batches
        self.val_losses.append(val_loss)
        
        # Log validation metrics
        self._log_metrics({'loss': val_loss}, epoch, prefix='val')
        
        # Generate validation samples
        if len(self.val_loader) > 0:
            sample_batch = next(iter(self.val_loader))
            text = sample_batch['texts'][0]
            ref_audio = sample_batch['mel_spectrograms'][0].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.forward(
                    text=text,
                    speaker_embedding=None,
                    emotion="neutral"
                )
                
                if outputs['waveform'] is not None:
                    # Log audio
                    self._log_audio_samples(
                        [outputs['waveform'][0]],
                        self.config.get('sampling_rate', 22050),
                        epoch,
                        prefix='val'
                    )
                    
                    # Save audio to file
                    audio_path = self.output_dir / f'val_sample_epoch_{epoch}.wav'
                    save_audio(
                        outputs['waveform'][0].cpu().numpy(),
                        str(audio_path),
                        self.config.get('sampling_rate', 22050)
                    )
        
        return val_loss
    
    def train(self):
        """Train the model"""
        print(f"Starting training for {self.args.epochs} epochs")
        
        # Loop over epochs
        for epoch in range(self.start_epoch, self.args.epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch+1}/{self.args.epochs} | Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(epoch)
            print(f"Epoch {epoch+1}/{self.args.epochs} | Val Loss: {val_loss:.4f}")
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                
            # Save model checkpoint
            if (epoch + 1) % self.args.save_interval == 0 or is_best:
                self._save_checkpoint(epoch, is_best)
            
            # Plot and save loss curves
            self._plot_losses()
        
        print("Training completed!")
        
        # Save final model
        self._save_checkpoint(self.args.epochs - 1, False)
        
        # Close tensorboard writer
        self.writer.close()
    
    def _plot_losses(self):
        """Plot and save loss curves"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_dir / 'loss_plot.png')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train voice cloning model")
    
    # Required arguments
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config file")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    
    # Optional arguments
    parser.add_argument("--dataset", type=str, default="vctk",
                       choices=["libri_tts", "vctk", "common_voice", "aishell3"],
                       help="Dataset name")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--speaker_encoder", type=str, default=None,
                       help="Path to pretrained speaker encoder")
    parser.add_argument("--vocoder", type=str, default=None,
                       help="Path to pretrained vocoder")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs to train")
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Logging interval (batches)")
    parser.add_argument("--save_interval", type=int, default=5,
                       help="Checkpoint saving interval (epochs)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--cpu", action="store_true",
                       help="Use CPU instead of GPU")
    
    args = parser.parse_args()
    
    # Initialize and start trainer
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()