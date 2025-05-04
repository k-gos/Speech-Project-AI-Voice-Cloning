import os
import sys
import requests
import hashlib
from pathlib import Path
from tqdm import tqdm

# Pretrained model URLs and checksums
PRETRAINED_MODELS = {
    "speaker_encoder": {
        "url": "https://github.com/resemble-ai/resemblyzer/raw/master/resemblyzer/pretrained.pt",
        "md5": "0cdba6f77e5fa2c40ebd0e3557944313",
        "description": "Speaker encoder model based on GE2E loss"
    },
    "hifigan_vocoder": {
        "url": "https://github.com/jik876/hifi-gan/releases/download/v1/g_02500000.pt",
        "md5": "1d25c1b1f064bd11f358d9c48e58da8f",
        "description": "HiFi-GAN universal vocoder trained on multiple datasets"
    }
}

def download_file(url, dest_path):
    """Download file from URL with progress bar"""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))
    
    return dest_path

def get_md5(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_pretrained_model(model_type, models_dir):
    """Download pretrained model if needed"""
    if model_type not in PRETRAINED_MODELS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Get model info
    model_info = PRETRAINED_MODELS[model_type]
    model_url = model_info["url"]
    model_md5 = model_info["md5"]
    
    # Determine filename from URL
    filename = os.path.basename(model_url)
    model_path = os.path.join(models_dir, filename)
    
    # Download if file doesn't exist or MD5 doesn't match
    if not os.path.exists(model_path) or get_md5(model_path) != model_md5:
        print(f"Downloading {model_type} model...")
        download_file(model_url, model_path)
        
        # Verify MD5
        if get_md5(model_path) != model_md5:
            raise ValueError(f"Downloaded model has incorrect MD5 hash")
    else:
        print(f"{model_type} model already exists and has correct MD5 hash")
    
    return model_path

def main():
    # Create directories
    os.makedirs("pretrained/speaker_encoder", exist_ok=True)
    os.makedirs("pretrained/vocoder", exist_ok=True)
    
    # Download models
    speaker_encoder_path = get_pretrained_model("speaker_encoder", "pretrained/speaker_encoder")
    print(f"Speaker encoder saved to: {speaker_encoder_path}")
    
    vocoder_path = get_pretrained_model("hifigan_vocoder", "pretrained/vocoder")
    print(f"Vocoder saved to: {vocoder_path}")
    
    print("\nDownload complete! Pretrained models are ready to use.")

if __name__ == "__main__":
    main()