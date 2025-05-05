import os
import sys
import argparse
import time
import json
import subprocess
from pathlib import Path
import multiprocessing
from tqdm import tqdm
import yaml
import logging
import datetime

def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"preprocess_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_preprocessing(args):
    """Run the main preprocessing script with the given arguments"""
    # Convert argparse Namespace to a dictionary for easier handling
    params = vars(args)
    
    # Setup logging
    log_file = setup_logging(args.log_dir)
    logging.info(f"Starting preprocessing of {args.dataset} dataset")
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    
    # Check if output directory exists and is not empty
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.force:
        logging.error("Output directory is not empty. Use --force to overwrite.")
        return False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create cache directory if using cache
    if args.use_cache:
        os.makedirs(args.cache_dir, exist_ok=True)
        cache_args = ["--cache-dir", args.cache_dir]
    else:
        cache_args = []
    
    # Load config file to get additional parameters
    try:
        config = load_config(args.config)
        logging.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logging.error(f"Error loading config file: {str(e)}")
        return False
    
    # Build command to run preprocessing
    cmd = [
        "python", "data/preprocess.py",
        "--dataset", args.dataset,
        "--input-dir", "script/"+args.input_dir,
        "--output-dir", args.output_dir,
        "--num-workers", str(args.num_workers),
        "--config", args.config,
        "--max-audio-len", str(args.max_audio_len)
    ]
    
    # Add cache arguments if using cache
    cmd.extend(cache_args)
    
    # Run preprocessing
    logging.info("Running preprocessing...")
    logging.info(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Read and log output in real-time
        for line in process.stdout:
            line = line.strip()
            if line:
                logging.info(line)
        
        process.wait()
        
        if process.returncode != 0:
            logging.error(f"Preprocessing failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        logging.error(f"Error running preprocessing script: {str(e)}")
        return False
        
    end_time = time.time()
    processing_time = end_time - start_time
    
    logging.info(f"Preprocessing completed in {processing_time:.2f} seconds")
    
    # Generate statistics
    try:
        logging.info("Generating dataset statistics...")
        
        # Count total audio files
        total_files = len(list(Path(args.output_dir).glob("**/*.wav")))
        logging.info(f"Total audio files: {total_files}")
        
        # Save dataset info
        dataset_info = {
            "dataset": args.dataset,
            "preprocessing_timestamp": datetime.datetime.now().isoformat(),
            "total_files": total_files,
            "max_audio_length": args.max_audio_len,
            "processing_time_seconds": processing_time
        }
        
        with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as f:
            json.dump(dataset_info, f, indent=2)
            
        logging.info(f"Dataset information saved to {os.path.join(args.output_dir, 'dataset_info.json')}")
        
    except Exception as e:
        logging.error(f"Error generating statistics: {str(e)}")
    
    logging.info(f"Preprocessing complete at {datetime.datetime.now().isoformat()}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for voice cloning training")
    
    parser.add_argument("-d", "--dataset", type=str, default="vctk",
                        choices=["vctk", "libri_tts", "common_voice", "aishell3", "ljspeech"],
                        help="Dataset to preprocess (default: vctk)")
    parser.add_argument("-i", "--input-dir", type=str, required=True,
                        help="Path to input dataset directory")
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Path to output directory")
    parser.add_argument("-w", "--num-workers", type=int, default=multiprocessing.cpu_count(),
                        help=f"Number of worker processes (default: {multiprocessing.cpu_count()})")
    parser.add_argument("-c", "--config", type=str, default="config/default_config.yaml",
                        help="Path to config file (default: config/default_config.yaml)")
    parser.add_argument("-l", "--log-dir", type=str, default="script/logs/preprocessing",
                        help="Directory for log files (default: script/logs/preprocessing)")
    parser.add_argument("-m", "--max-audio-len", type=float, default=10.0,
                        help="Maximum audio length in seconds (default: 10.0)")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false",
                        help="Disable caching of intermediate results")
    parser.add_argument("--cache-dir", type=str, default="cache",
                        help="Directory for cached files (default: cache)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force overwrite existing output directory")
    
    parser.set_defaults(use_cache=True)
    
    args = parser.parse_args()
    
    print(args)
    # Run preprocessing
    success = run_preprocessing(args)
    
    if success:
        print("\nPreprocessing completed successfully.")
        print(f"Dataset saved to {args.output_dir}")
        print(f"Check logs for details.")
    else:
        print("\nPreprocessing failed. Check logs for errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()