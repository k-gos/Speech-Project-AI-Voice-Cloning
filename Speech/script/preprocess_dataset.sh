#!/bin/bash
# Script to preprocess datasets for voice cloning

# Default settings
DATASET="vctk"
INPUT_DIR=""
OUTPUT_DIR=""
NUM_WORKERS=4
CONFIG_FILE="config/default_config.yaml"
LOG_DIR="logs/preprocessing"
MAX_AUDIO_LEN=10
CACHE_DIR="cache"
USE_CACHE=true
FORCE=false

# Display help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Preprocess dataset for voice cloning training"
    echo ""
    echo "Options:"
    echo "  -d, --dataset DATASET     Dataset to preprocess (vctk, libri_tts, common_voice, aishell3, ljspeech)"
    echo "  -i, --input-dir DIR       Path to input dataset directory"
    echo "  -o, --output-dir DIR      Path to output directory"
    echo "  -w, --workers NUM         Number of worker processes (default: 4)"
    echo "  -c, --config FILE         Path to config file (default: config/default_config.yaml)"
    echo "  -l, --log-dir DIR         Directory for log files (default: logs/preprocessing)"
    echo "  -m, --max-audio-len SEC   Maximum audio length in seconds (default: 10)"
    echo "      --no-cache            Disable caching of intermediate results"
    echo "      --cache-dir DIR       Directory for cached files (default: cache)"
    echo "  -f, --force               Force overwrite existing output directory"
    echo "  -h, --help                Show this help message"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -i|--input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -w|--workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -l|--log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -m|--max-audio-len)
            MAX_AUDIO_LEN="$2"
            shift 2
            ;;
        --no-cache)
            USE_CACHE=false
            shift
            ;;
        --cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$INPUT_DIR" ]; then
    echo "Input directory is required"
    show_help
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Output directory is required"
    show_help
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/preprocess_${DATASET}_${TIMESTAMP}.log"

# Check if output directory exists and is not empty
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR")" ] && [ "$FORCE" = false ]; then
    echo "Output directory is not empty. Use --force to overwrite."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create cache directory if using cache
if [ "$USE_CACHE" = true ]; then
    mkdir -p "$CACHE_DIR"
    CACHE_ARGS="--cache-dir $CACHE_DIR"
else
    CACHE_ARGS=""
fi

echo "Starting preprocessing of $DATASET dataset at $(date)"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Logging to: $LOG_FILE"

# Run preprocessing
echo "Running preprocessing with Python script..."
{
    python data/preprocess.py \
        --dataset "$DATASET" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --num-workers "$NUM_WORKERS" \
        --config "$CONFIG_FILE" \
        --max-audio-len "$MAX_AUDIO_LEN" \
        $CACHE_ARGS \
        2>&1
} | tee "$LOG_FILE"

# Check if preprocessing was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Preprocessing completed successfully."
    echo "Dataset saved to $OUTPUT_DIR"
    echo "Stats and metadata logged to $LOG_FILE"
else
    echo "Preprocessing failed. Check $LOG_FILE for errors."
    exit 1
fi

# Generate dataset statistics
echo "Generating dataset statistics..."

# Calculate total audio duration and other statistics
TOTAL_FILES=$(find "$OUTPUT_DIR" -name "*.wav" | wc -l)
echo "Total audio files: $TOTAL_FILES" | tee -a "$LOG_FILE"

# Save dataset info
{
    echo "Dataset: $DATASET"
    echo "Preprocessing timestamp: $(date)"
    echo "Total files: $TOTAL_FILES"
    echo "Max audio length: $MAX_AUDIO_LEN seconds"
} > "$OUTPUT_DIR/dataset_info.txt"

echo "Preprocessing complete at $(date)"