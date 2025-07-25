#!/bin/bash




SOURCE_ZIP_FILE="dsec-det_left_object_detections.zip"

# 2. Path to the base directory on your filesystem.
FILESYSTEM_BASE_PATH="../data/DSEC_Night/"
# --- END OF CONFIGURATION ---

INNER_ZIP_NAME="train_object_detections.zip"


wget https://download.ifi.uzh.ch/rpg/DSEC/detection/dsec-det_left_object_detections.zip

# 1. Validate paths
if [ ! -f "$SOURCE_ZIP_FILE" ]; then
    echo "Error: Source ZIP file not found at '$SOURCE_ZIP_FILE'"
    exit 1
fi
if [ ! -d "$FILESYSTEM_BASE_PATH" ]; then
    echo "Error: Filesystem base path not found at '$FILESYSTEM_BASE_PATH'"
    exit 1
fi



# 2. Create temporary directories and set a trap to clean them up on exit
TEMP_OUTER=$(mktemp -d)
TEMP_INNER=$(mktemp -d)
trap 'rm -rf "$TEMP_OUTER" "$TEMP_INNER"' EXIT

echo "Extracting to find '$INNER_ZIP_NAME'..."
# 3. Extract only the inner zip file into the first temp directory
unzip -q "$SOURCE_ZIP_FILE" "$INNER_ZIP_NAME" -d "$TEMP_OUTER"
if [ $? -ne 0 ]; then
    echo "Error: Could not extract '$INNER_ZIP_NAME' from the main ZIP. Is it present?"
    exit 1
fi

echo "Extracting '$INNER_ZIP_NAME' contents..."
# 4. Extract the contents of the inner zip into the second temp directory
unzip -q "$TEMP_OUTER/$INNER_ZIP_NAME" -d "$TEMP_INNER"

EXTRACTED_TRAIN_PATH="$TEMP_INNER/train"
if [ ! -d "$EXTRACTED_TRAIN_PATH" ]; then
    echo "Error: 'train' directory not found inside '$INNER_ZIP_NAME'."
    exit 1
fi

echo ""
echo "Checking filesystem and moving files..."
# 5. Loop through all subfolders inside the extracted 'train' directory
for source_folder_path in "$EXTRACTED_TRAIN_PATH"/*; do
    if [ -d "$source_folder_path" ]; then
        folder_name=$(basename "$source_folder_path")
        filesystem_folder_path="$FILESYSTEM_BASE_PATH/$folder_name"

        # Check if the corresponding folder exists on the filesystem
        if [ -d "$filesystem_folder_path" ]; then
            source_od_path="$source_folder_path/object_detections"
            dest_od_path="$filesystem_folder_path/object_detections"

            if [ -d "$source_od_path" ]; then
                # Ensure destination exists and move all files
                mkdir -p "$dest_od_path"
                mv "$source_od_path"/* "$dest_od_path"/
                echo "Unified files for '$folder_name'"
            else
                echo "Note: No 'object_detections' folder found in ZIP for '$folder_name'."
            fi
        else
            echo "Ignoring '$folder_name' (not found in filesystem)."
        fi
    fi
done

rm -rf dsec-det_left_object_detections.zip
echo ""
echo "Process finished."
