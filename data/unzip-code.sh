#!/bin/bash

SOURCE_DIR="/kaggle/input/data-science-bowl-2018"
TARGET_DIR="/kaggle/working/dsb-data-2018"

mkdir -p "$TARGET_DIR"

# 3. Loop through all .zip files in the source directory
for zip_file in "$SOURCE_DIR"/*.zip; do
    
    # Check if any zip files actually exist to avoid errors
    [ -e "$zip_file" ] || continue
    
    echo "Extracting: $zip_file"
    
    # -d specifies the destination directory
    # -q runs 'quietly' to keep your terminal clean
    unzip -q "$zip_file" -d "$TARGET_DIR"
done

echo "Done! All files moved to $TARGET_DIR"