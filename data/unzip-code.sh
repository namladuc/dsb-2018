#!/bin/bash

SOURCE_DIR="/kaggle/input/data-science-bowl-2018"
TARGET_DIR="/kaggle/working/dsb-data-2018"

mkdir -p "$TARGET_DIR"

# 3. Loop through all .zip files in the source directory
for zip_file in "$SOURCE_DIR"/*.zip; do
    [ -e "$zip_file" ] || continue

    # 1. Get the filename without the path (e.g., "data.zip")
    base_name=$(basename "$zip_file")

    # 2. Strip the .zip extension (e.g., "data")
    folder_name="${base_name%.*}"

    # 3. Define the specific path for this zip
    extraction_path="$TARGET_DIR/$folder_name"

    echo "Extracting $base_name to $extraction_path..."

    # 4. Create the specific subfolder and unzip into it
    mkdir -p "$extraction_path"
    unzip -q "$zip_file" -d "$extraction_path"
done

echo "Done! All files moved to $TARGET_DIR"
