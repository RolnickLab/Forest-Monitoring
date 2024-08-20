#!/bin/bash

# URL of the tar file
URL="https://zenodo.org/records/13126114/files/forest_monitoring.tar.gz?download=1"

# Name of the downloaded file
TAR_FILE="forest_monitoring.tar.gz"

# Target folder for extraction
TARGET_FOLDER="data/tree_data"

# Download the tar file
echo "Downloading $TAR_FILE..."
wget -O "$TAR_FILE" "$URL"

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Download failed. Exiting."
    exit 1
fi

# Create target folder
echo "Creating target folder: $TARGET_FOLDER"
mkdir -p "$TARGET_FOLDER"

# Extract to target folder
echo "Extracting $TAR_FILE to $TARGET_FOLDER..."
tar -xzf "$TAR_FILE" -C "$TARGET_FOLDER"

# Check if extraction was successful
if [ $? -ne 0 ]; then
    echo "Extraction failed. Exiting."
    exit 1
fi

# Delete the tar file
echo "Deleting $TAR_FILE..."
rm "$TAR_FILE"

echo "Process completed successfully."