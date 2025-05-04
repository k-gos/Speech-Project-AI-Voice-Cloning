"""
Text processing utilities for voice cloning system
"""
import re
import torch
import unicodedata
from typing import List, Dict, Optional
import string
import nltk
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class TextProcessor:
    """Text processor for preparing text inputs"""
    
    def __init__(self, use_bert: bool = True, bert_model: str = "bert-base-uncased"):
        """
        Initialize text processor
        
        Args:
            use_bert: Whether to use BERT tokenizer
            bert_model: BERT model name to use
        """
        self.use_bert = use_bert
        
        if use_bert:
            self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        
        # Basic punctuation for simple processing
        self.punctuation = '!,.;:?'
        self.whitespace_pattern = re.compile(r'\s+')
        
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by cleaning and standardizing
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        
        # Replace multiple whitespaces with single space
        text = self.whitespace_pattern.sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> Dict:
        """
        Tokenize text for model input
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with tokenized output
        """
        if self.use_bert:
            # Use BERT tokenizer
            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "tokens": self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
            }
        else:
            # Simple tokenization
            words = word_tokenize(text)
            return {"tokens": words}
            
    def add_punctuation(self, text: str, add_period: bool = True) -> str:
        """
        Add punctuation to text if missing
        
        Args:
            text: Input text
            add_period: Whether to add period at end if no punctuation
            
        Returns:
            Text with punctuation
        """
        text = text.strip()
        
        if text and add_period:
            if text[-1] not in self.punctuation:
                text = text + '.'
                
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Use NLTK's sentence tokenizer
        from nltk.tokenize import sent_tokenize
        
        # Make sure text ends with punctuation
        text = self.add_punctuation(text)
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        return sentences


def prepare_text_for_tts(text: str) -> List[str]:
    """
    Prepare text for TTS by splitting into manageable chunks
    
    Args:
        text: Input text
        
    Returns:
        List of text chunks ready for TTS
    """
    processor = TextProcessor()
    
    # Normalize text
    text = processor.normalize_text(text)
    
    # Split into sentences
    sentences = processor.split_into_sentences(text)
    
    # Group sentences into chunks of reasonable size
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would make chunk too long, start new chunk
        if len(current_chunk) + len(sentence) > 250 and current_chunk:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
                
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks


def clean_text_for_filename(text: str, max_length: int = 50) -> str:
    """
    Clean text to be used as a filename
    
    Args:
        text: Input text
        max_length: Maximum filename length
        
    Returns:
        Clean filename-safe text
    """
    # Remove non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Replace spaces with underscores
    text = re.sub(r'\s+', '_', text)
    
    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length]
        
    return text.lower()


if __name__ == "__main__":
    # Test the text processor
    processor = TextProcessor()
    
    test_text = "Hello, world! This is a test. How are you doing today?"
    
    print("Original text:", test_text)
    print("Normalized text:", processor.normalize_text(test_text))
    
    tokenized = processor.tokenize(test_text)
    print("Tokenized:", tokenized["tokens"])
    
    sentences = processor.split_into_sentences(test_text)
    print("Sentences:", sentences)
    
    chunks = prepare_text_for_tts("This is a longer text that should be split into multiple chunks. "
                                 "It contains several sentences of varying length. "
                                 "Some are short. Others are much longer and more complex, "
                                 "containing multiple clauses and phrases.")
    print("TTS chunks:", chunks)