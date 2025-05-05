import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    """Text encoder using pretrained transformer model"""
    
    def __init__(self, model_name="bert-base-uncased", output_dim=512, max_text_length=200):
        super(TextEncoder, self).__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.max_text_length = max_text_length
        
        # Load pretrained tokenizer and model
        print(f"Loading pretrained text encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Projection layer if needed
        self.projection = None
        if self.model.config.hidden_size != output_dim:
            self.projection = nn.Linear(self.model.config.hidden_size, output_dim)
    
    def forward(self, text_list):
        """
        Encode text strings to embeddings
        
        Args:
            text_list: List of text strings
            
        Returns:
            text_memory: Encoded text (batch_size, seq_len, output_dim)
        """
        # Tokenize
        encoded_dict = self.tokenizer(
            text_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        
        # Move to same device as model
        input_ids = encoded_dict['input_ids'].to(self.model.device)
        attention_mask = encoded_dict['attention_mask'].to(self.model.device)
        
        # Get BERT embeddings
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get hidden states from last layer
        hidden_states = outputs.last_hidden_state
        
        # Project if needed
        if self.projection is not None:
            hidden_states = self.projection(hidden_states)
        
        return hidden_states