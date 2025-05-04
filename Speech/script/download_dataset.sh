#!/bin/bash
# Script to download voice cloning datasets

# Set default directories
DATASETS_DIR="datasets"
TEMP_DIR="/tmp/voice_cloning_downloads"

# Print header
print_header() {
    echo "======================================================================"
    echo "  Voice Cloning Dataset Downloader"
    echo "======================================================================"
    echo ""
}

# Print usage
print_usage() {
    echo "Usage: $0 [options] dataset_name"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR   Directory to save datasets (default: ./datasets)"
    echo "  --temp-dir DIR     Directory for temporary files (default: /tmp/voice_cloning_downloads)"
    echo "  --help             Show this help message"
    echo ""
    echo "Available datasets:"
    echo "  vctk              VCTK Corpus (English, multi-speaker)"
    echo "  ljspeech          LJSpeech (English, single female speaker)"
    echo "  libritts          LibriTTS (English, multi-speaker)"
    echo "  common_voice      Mozilla Common Voice (multilingual)"
    echo "  aishell3          AISHELL-3 (Mandarin Chinese, multi-speaker)"
    echo ""
}

# Download VCTK dataset
download_vctk() {
    echo "Downloading VCTK dataset..."
    
    # Create directory
    mkdir -p "$DATASETS_DIR/vctk"
    
    # Download
    VCTK_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
    VCTK_ZIP="$TEMP_DIR/vctk.zip"
    
    wget -O "$VCTK_ZIP" "$VCTK_URL" || { echo "Failed to download VCTK"; exit 1; }
    
    # Extract
    echo "Extracting VCTK dataset..."
    unzip -q "$VCTK_ZIP" -d "$TEMP_DIR"
    
    # Move to datasets directory
    mv "$TEMP_DIR/VCTK-Corpus-0.92"/* "$DATASETS_DIR/vctk/"
    
    # Clean up
    rm "$VCTK_ZIP"
    
    echo "VCTK dataset downloaded to $DATASETS_DIR/vctk"
}

# Download LJSpeech dataset
download_ljspeech() {
    echo "Downloading LJSpeech dataset..."
    
    # Create directory
    mkdir -p "$DATASETS_DIR/ljspeech"
    
    # Download
    LJSPEECH_URL="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    LJSPEECH_TAR="$TEMP_DIR/ljspeech.tar.bz2"
    
    wget -O "$LJSPEECH_TAR" "$LJSPEECH_URL" || { echo "Failed to download LJSpeech"; exit 1; }
    
    # Extract
    echo "Extracting LJSpeech dataset..."
    tar -xjf "$LJSPEECH_TAR" -C "$TEMP_DIR"
    
    # Move to datasets directory
    mv "$TEMP_DIR/LJSpeech-1.1"/* "$DATASETS_DIR/ljspeech/"
    
    # Clean up
    rm "$LJSPEECH_TAR"
    
    echo "LJSpeech dataset downloaded to $DATASETS_DIR/ljspeech"
}

# Download LibriTTS dataset
download_libritts() {
    echo "Downloading LibriTTS dataset..."
    
    # Create directory
    mkdir -p "$DATASETS_DIR/libritts"
    
    # List of LibriTTS parts to download
    PARTS=("dev-clean" "test-clean" "train-clean-100")
    
    for PART in "${PARTS[@]}"; do
        echo "Downloading LibriTTS $PART..."
        URL="https://www.openslr.org/resources/60/LibriTTS/$PART.tar.gz"
        TAR_FILE="$TEMP_DIR/$PART.tar.gz"
        
        wget -O "$TAR_FILE" "$URL" || { echo "Failed to download LibriTTS $PART"; continue; }
        
        # Extract
        echo "Extracting LibriTTS $PART..."
        tar -xzf "$TAR_FILE" -C "$TEMP_DIR"
        
        # Move to datasets directory
        mv "$TEMP_DIR/LibriTTS/$PART" "$DATASETS_DIR/libritts/"
        
        # Clean up
        rm "$TAR_FILE"
        
        echo "LibriTTS $PART downloaded"
    done
    
    echo "LibriTTS dataset downloaded to $DATASETS_DIR/libritts"
}

# Download Common Voice dataset
download_common_voice() {
    echo "Common Voice dataset needs to be downloaded manually due to license restrictions."
    echo ""
    echo "Please follow these steps:"
    echo "1. Visit https://commonvoice.mozilla.org/datasets"
    echo "2. Create an account and agree to the terms"
    echo "3. Download the language version you want (e.g., English)"
    echo "4. Extract the downloaded tarball"
    echo "5. Move the extracted files to $DATASETS_DIR/common_voice"
    echo ""
}

# Download AISHELL-3 dataset
download_aishell3() {
    echo "AISHELL-3 dataset needs to be downloaded manually due to license restrictions."
    echo ""
    echo "Please follow these steps:"
    echo "1. Visit https://www.openslr.org/93/"
    echo "2. Download the dataset files"
    echo "3. Extract the downloaded files"
    echo "4. Move the extracted files to $DATASETS_DIR/aishell3"
    echo ""
    echo "Note: AISHELL-3 is a Mandarin Chinese speech corpus."
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            DATASETS_DIR="$2"
            shift 2
            ;;
        --temp-dir)
            TEMP_DIR="$2"
            shift 2
            ;;
        --help)
            print_header
            print_usage
            exit 0
            ;;
        vctk|ljspeech|libritts|common_voice|aishell3)
            DATASET="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check if dataset is specified
if [ -z "$DATASET" ]; then
    print_header
    print_usage
    exit 1
fi

# Create directories
mkdir -p "$DATASETS_DIR"
mkdir -p "$TEMP_DIR"

# Download selected dataset
print_header
echo "Downloading $DATASET dataset to $DATASETS_DIR"
echo ""

case $DATASET in
    vctk)
        download_vctk
        ;;
    ljspeech)
        download_ljspeech
        ;;
    libritts)
        download_libritts
        ;;
    common_voice)
        download_common_voice
        ;;
    aishell3)
        download_aishell3
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        print_usage
        exit 1
        ;;
esac

# Clean up temporary directory
rm -rf "$TEMP_DIR"

echo ""
echo "Download process completed!"
echo "Next steps:"
echo "1. Preprocess the dataset: python src/data/preprocess.py --dataset $DATASET --input_dir $DATASETS_DIR/$DATASET --output_dir $DATASETS_DIR/${DATASET}_processed"
echo "2. Start training: python src/train.py --config config/default_config.yaml --data_dir $DATASETS_DIR/${DATASET}_processed --output_dir outputs/$DATASET"