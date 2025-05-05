import os
import sys
import requests
import hashlib
import gdown
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
        # Use gdown for Google Drive downloads
        "gdrive_id": "1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y",
        "folder_name": "UNIVERSAL_V1",  # Based on the screenshot
        "file_name": "g_02500000.pt",   # Standard filename for HiFi-GAN checkpoint
        "description": "HiFi-GAN universal vocoder from Google Drive"
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

def download_gdrive_file(gdrive_id, folder_name, file_name, dest_path):
    """Download file from Google Drive"""
    try:
        # Install gdown if not already installed
        try:
            import gdown
        except ImportError:
            print("Installing gdown for Google Drive downloads...")
            os.system(f"{sys.executable} -m pip install gdown")
            import gdown
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Construct the direct file URL if possible
        # Format: https://drive.google.com/uc?id=FILE_ID
        file_url = f"https://drive.google.com/uc?id={gdrive_id}"
        
        print(f"Downloading {file_name} from Google Drive folder {folder_name}...")
        gdown.download(file_url, dest_path, quiet=False)
        
        # If file is not found, try downloading the entire folder
        if not os.path.exists(dest_path):
            print(f"Direct file download failed. Attempting to download the folder...")
            folder_url = f"https://drive.google.com/drive/folders/{gdrive_id}"
            output_dir = os.path.dirname(dest_path)
            gdown.download_folder(folder_url, output=output_dir, quiet=False)
            
            # Find the model file in the downloaded folder
            model_path = None
            for root, dirs, files in os.walk(output_dir):
                if file_name in files:
                    model_path = os.path.join(root, file_name)
                    break
            
            if model_path:
                # Copy the file to the destination
                import shutil
                shutil.copy(model_path, dest_path)
                print(f"Found and copied model file to {dest_path}")
            else:
                raise FileNotFoundError(f"Could not find {file_name} in downloaded folder")
        
        return dest_path
    
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        print("\nManual download instructions:")
        print(f"1. Visit: https://drive.google.com/drive/folders/{gdrive_id}")
        print(f"2. Navigate to the {folder_name} folder")
        print(f"3. Download the {file_name} file")
        print(f"4. Save it to: {dest_path}")
        return None

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
    
    # Handle Google Drive downloads differently
    if "gdrive_id" in model_info:
        gdrive_id = model_info["gdrive_id"]
        folder_name = model_info.get("folder_name", "")
        file_name = model_info.get("file_name", "g_02500000.pt")
        
        model_path = os.path.join(models_dir, file_name)
        
        # Download if file doesn't exist
        if not os.path.exists(model_path):
            print(f"Downloading {model_type} model from Google Drive...")
            return download_gdrive_file(gdrive_id, folder_name, file_name, model_path)
        else:
            print(f"{model_type} model already exists at {model_path}")
            return model_path
    else:
        # Regular URL download
        model_url = model_info["url"]
        model_md5 = model_info.get("md5", None)
        
        # Determine filename from URL
        filename = os.path.basename(model_url)
        model_path = os.path.join(models_dir, filename)
        
        # Download if file doesn't exist or MD5 doesn't match
        if not os.path.exists(model_path) or (model_md5 and get_md5(model_path) != model_md5):
            print(f"Downloading {model_type} model...")
            download_file(model_url, model_path)
            
            # Verify MD5 if provided
            if model_md5 and get_md5(model_path) != model_md5:
                print(f"Warning: Downloaded model has incorrect MD5 hash")
                print(f"Expected: {model_md5}")
                print(f"Actual: {get_md5(model_path)}")
        else:
            print(f"{model_type} model already exists at {model_path}")
        
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
    
    # Additional manual instructions if download fails
    if not vocoder_path:
        print("\nIf automatic download fails, please follow these manual steps:")
        print("1. Download the HiFi-GAN model from:")
        print("   https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y")
        print("2. Look for the UNIVERSAL_V1 folder")
        print("3. Download the g_02500000.pt file")
        print("4. Create the directory: pretrained/vocoder")
        print("5. Place the downloaded file in this directory")

if __name__ == "__main__":
    main()