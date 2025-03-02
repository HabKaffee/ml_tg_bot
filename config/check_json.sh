#!/bin/bash

validate_json_files() {
    local folder="$1"
    echo "Validating JSON files in folder: $folder"

    for file in $(find "$folder" -name "*.json"); do
        echo "Checking $file ..."
        jq empty "$file" || exit 1
    done

    echo "All JSON files in $folder are valid."
}

if [ -z "$1" ]; then
    echo "Usage: $0 <folder-to-analyze>"
    exit 1
fi

validate_json_files "$1"
