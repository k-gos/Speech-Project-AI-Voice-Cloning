import torch
import torch.nn as nn
from typing import List

class SimpleTextEncoder(nn.Module):
    """Simple text encoder using embedding and LSTM layers"""
    
    def __init__(self, vocab_size=10000, embedding_dim=256, hidden_dim=512, output_dim=512, max_text_length=200):
        super(SimpleTextEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_text_length = max_text_length
        
        # Character embedding
        self.char_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Projection to output dimension
        self.projection = nn.Linear(hidden_dim * 2, output_dim)
        
        # Character to index mapping
        self.char_to_index = {chr(i+32): i for i in range(95)}  # ASCII 32-126
        self.char_to_index['<pad>'] = len(self.char_to_index)
        self.char_to_index['<unk>'] = len(self.char_to_index)
    
    def _text_to_indices(self, text_list):
        """Convert text strings to index tensors"""
        batch_indices = []
        
        for text in text_list:
            # Convert characters to indices
            indices = [self.char_to_index.get(c, self.char_to_index['<unk>']) for c in text[:self.max_text_length]]
            
            # Pad if necessary
            if len(indices) < self.max_text_length:
                indices += [self.char_to_index['<pad>']] * (self.max_text_length - len(indices))
                
            batch_indices.append(indices)
            
        return torch.tensor(batch_indices, dtype=torch.long)
    
    def forward(self, text_list: List[str]):
        """
        Encode text strings to embeddings
        
        Args:
            text_list: List of text strings
            
        Returns:
            text_memory: Encoded text (batch_size, seq_len, output_dim)
        """
        # Convert text to indices
        indices = self._text_to_indices(text_list).to(next(self.parameters()).device)
        
        # Embed characters
        embedded = self.char_embedding(indices)
        
        # Pass through LSTM
        outputs, _ = self.lstm(embedded)
        
        # Project to output dimension
        projected = self.projection(outputs)
        
        return projected


class TextEncoder(nn.Module):
    """
    Text encoder using a simpler architecture to avoid Hugging Face dependencies
    """
    def __init__(self, model_name=None, output_dim=512, max_text_length=200):
        super(TextEncoder, self).__init__()
        
        # Ignore model_name and use SimpleTextEncoder instead
        print("Using SimpleTextEncoder instead of transformers due to compatibility issues")
        self.encoder = SimpleTextEncoder(
            vocab_size=10000,
            embedding_dim=256,
            hidden_dim=512,
            output_dim=output_dim,
            max_text_length=max_text_length
        )
    
    def forward(self, text_list):
        """Forward pass"""
        return self.encoder(text_list)