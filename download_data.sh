#!/bin/bash

# Check if gdown is installed
if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown is not installed. Please install it using 'pip install gdown'"
  exit 1
fi

# Create the destination folder if it does not exist
DATA_DIR="./data"
mkdir -p "$DATA_DIR"

# List of Google Drive file IDs
FILE_IDS=(
  "1-R-XtkEN3-pNNdMPUkWyocrq3zZc2W-5"
  "1-P_QXP0Rbk-_WMwZQ8X5HgTsVHMNHnRS"
  "1-MYjC7QmtPzvr83Iy6Pvehuk45L7gqAv"
  "1-8EhjAD_lEqU4dX_ST2G9LNc4sygrVlc"
  "1-6ioOM5qFZruL742b1Ou8bIvpwzkk405"
  "1-4_o29jv7q-JUklx3nXmHNt5RaVK7e7N"
  "1-2LB2RUvJpMcZXa5pLo0wtdv2MWWGp_d"
  "1--NY0SagTnlCojuqun-Fg6Lp2hYeRF29"
  "15IjYhOXP25tOIwmUlsUAbMcnM8lfMGzX"
)

# Download each file into the destination folder
for FILE_ID in "${FILE_IDS[@]}"; do
  OUTPUT_FILE="$DATA_DIR/$(gdown --id "$FILE_ID" --print-url | awk -F '/' '{print $NF}')"
  gdown "$FILE_ID" -O "$OUTPUT_FILE"
done