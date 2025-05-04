import os
import argparse
import subprocess
import sys
from pathlib import Path
import shutil
import requests
import tarfile
import zipfile
from tqdm import tqdm
import time

# Dataset configurations
DATASETS = {
    "vctk": {
        "url": "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip",
        "extract_dir": "VCTK-Corpus",
        "description": "VCTK Corpus - 110 English speakers with various accents",
        "size_mb": 11000  # ~11GB
    },
    "libri_tts": {
        "url": "http://www.openslr.org/resources/60/train-clean-100.tar.gz",
        "extract_dir": "LibriTTS",
        "description": "LibriTTS train-clean-100 subset - 247 speakers",
        "size_mb": 6000  # ~6GB
    },
    "ljspeech": {
        "url": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        "extract_dir": "LJSpeech-1.1",
        "description": "LJSpeech - Single female speaker dataset",
        "size_mb": 2500  # ~2.5GB
    },
    "common_voice": {
        "url": "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-6.1-2020-12-11/en.tar.gz",
        "extract_dir": "common_voice",
        "description": "Mozilla Common Voice English dataset",
        "size_mb": 12000  # ~12GB
    },
    "aishell3": {
        "url": "None",  # Requires registration
        "extract_dir": "aishell3",
        "description": "AISHELL-3 Mandarin dataset (requires manual download from OpenSLR)",
        "size_mb": 18000  # ~18GB
    }
}

def show_help():
    """Display help message"""
    print("Download datasets for voice cloning training")
    print("\nUsage: python download_datasets.py [dataset_name] [options]")
    print("\nAvailable datasets:")
    
    for name, info in DATASETS.items():
        print(f"  {name}: {info['description']} (~{info['size_mb']//1000}GB)")
    
    print("\nOptions:")
    print("  --output-dir DIR   Directory to save datasets (default: datasets)")
    print("  --no-extract       Download only, don't extract archives")
    print("  --help             Show this help message")

def download_file(url, dest_path):
    """Download file from URL with progress bar"""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB
        
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
        
        # Wait a moment to ensure file is fully written
        time.sleep(1)
        
        return dest_path
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {str(e)}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise

def extract_archive(archive_path, extract_dir):
    """Extract archive file (zip, tar.gz, tar.bz2)"""
    print(f"Extracting {os.path.basename(archive_path)} to {extract_dir}...")
    
    os.makedirs(extract_dir, exist_ok=True)
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            # Get total size for progress bar
            total_size = sum(info.file_size for info in zip_ref.infolist())
            
            # Extract with progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                for file in zip_ref.infolist():
                    zip_ref.extract(file, extract_dir)
                    pbar.update(file.file_size)
    
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            # Get members for progress bar
            members = tar.getmembers()
            
            # Extract with progress bar
            with tqdm(total=len(members), desc="Extracting") as pbar:
                for member in members:
                    tar.extract(member, extract_dir)
                    pbar.update(1)
    
    elif archive_path.endswith('.tar.bz2'):
        with tarfile.open(archive_path, 'r:bz2') as tar:
            # Get members for progress bar
            members = tar.getmembers()
            
            # Extract with progress bar
            with tqdm(total=len(members), desc="Extracting") as pbar:
                for member in members:
                    tar.extract(member, extract_dir)
                    pbar.update(1)
    
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

def download_dataset(dataset_name, output_dir, extract=True):
    """Download and optionally extract a dataset"""
    if dataset_name not in DATASETS:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print("Available datasets: " + ", ".join(DATASETS.keys()))
        return False
    
    dataset_info = DATASETS[dataset_name]
    url = dataset_info["url"]
    extract_dir = dataset_info["extract_dir"]
    
    # Check if URL is available
    if url == "None" or url is None:
        print(f"Error: {dataset_name} requires manual download.")
        if dataset_name == "aishell3":
            print("Please download AISHELL-3 from http://www.openslr.org/93/")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Archive path
    archive_name = url.split('/')[-1]
    archive_path = os.path.join(output_dir, archive_name)
    
    # Final extraction path
    extract_path = os.path.join(output_dir, extract_dir)
    
    # Check if dataset already exists
    if os.path.exists(extract_path) and os.listdir(extract_path):
        print(f"Dataset {dataset_name} already exists at {extract_path}")
        return True
    
    # Download dataset
    print(f"Downloading {dataset_name} dataset (~{dataset_info['size_mb']//1000}GB)...")
    try:
        download_file(url, archive_path)
    except Exception as e:
        print(f"Error downloading {dataset_name}: {str(e)}")
        return False
    
    # Extract dataset if requested
    if extract:
        try:
            extract_archive(archive_path, output_dir)
            print(f"Successfully extracted {dataset_name} to {extract_path}")
            
            # Clean up archive if extraction successful
            os.remove(archive_path)
            print(f"Removed archive file {archive_path}")
        except Exception as e:
            print(f"Error extracting {dataset_name}: {str(e)}")
            return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Download datasets for voice cloning")
    parser.add_argument("dataset", nargs='?', help="Dataset name to download (vctk, libri_tts, ljspeech, common_voice, aishell3)")
    parser.add_argument("--output-dir", default="datasets", help="Directory to save datasets (default: datasets)")
    parser.add_argument("--no-extract", action="store_true", help="Download only, don't extract archives")
    args = parser.parse_args()
    
    # Show help if no arguments provided
    if not args.dataset:
        show_help()
        return
    
    # Download the selected dataset
    success = download_dataset(args.dataset, args.output_dir, extract=not args.no_extract)
    
    if success:
        print(f"\nDataset {args.dataset} is ready at {os.path.join(args.output_dir, DATASETS[args.dataset]['extract_dir'])}")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()