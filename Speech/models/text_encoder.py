"""
Text Encoder Module

Encodes input text into embeddings for speech synthesis using transformer-based models.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    """
    Text encoder that converts text to embeddings using transformer models
    """
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 output_dim: int = 512,
                 max_text_length: int = 200,
                 device: Optional[torch.device] = None):
        """
        Initialize the text encoder
        
        Args:
            model_name: Pretrained transformer model name
            output_dim: Dimension of output embeddings
            max_text_length: Maximum text length for processing
            device: Device to run the model on
        """
        super().__init__()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.max_text_length = max_text_length
        
        # Load pretrained tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze transformer parameters to reduce training time and memory
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        # Projection layer to map transformer output dimension to required output dimension
        self.projection = nn.Linear(self.transformer.config.hidden_size, output_dim)
        
        # Move model to device
        self.to(device)
        print(f"Text encoder initialized with {model_name} on {device}")
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text into embeddings
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Text embeddings (shape: [batch_size, seq_length, output_dim])
        """
        # Tokenize text
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Get transformer outputs (without computing gradients to save memory)
        with torch.no_grad():
            outputs = self.transformer(**inputs)
            
        # Get the hidden states
        hidden_states = outputs.last_hidden_state
        
        # Project to output dimension
        embeddings = self.projection(hidden_states)
        
        return embeddings
    
    def get_phoneme_embeddings(self, phoneme_sequences: List[List[str]]) -> torch.Tensor:
        """
        Get embeddings for phoneme sequences (for more precise pronunciation control)
        
        Args:
            phoneme_sequences: List of phoneme sequences
            
        Returns:
            Phoneme embeddings
        """
        # Convert phoneme sequences to strings that the tokenizer can process
        phoneme_texts = [" ".join(phonemes) for phonemes in phoneme_sequences]
        return self.forward(phoneme_texts)