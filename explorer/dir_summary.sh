#!/bin/bash

# Function to summarize a directory
summarize_dir() {
    local dir="$1"
    local depth="$2"
    local max_depth="$3"
    local max_items="$4"  # Maximum number of items (files + dirs) before skipping
    local content_ratio="$5"  # Ratio of file content to show
    local max_lines="$6"  # Max number of lines to show from each file

    # Check if max depth reached
    if [ "$depth" -gt "$max_depth" ]; then
        echo "Skipping directory ${dir%/} (Max depth reached)"
        return
    fi

    # Get list of items in the directory
    local items=("$dir"/*)
    local item_count=${#items[@]}
    local displayed_count=0

    # Loop through items
    for item in "${items[@]}"; do
        # Limit the number of items displayed
        if [ "$displayed_count" -ge "$max_items" ]; then
            echo "Skipping remaining items in ${dir%/} (Too many items)"
            break
        fi

        # Check if item is a directory
        if [ -d "$item" ]; then
            echo "Skipping directory ${item%/} (Max depth reached)"
            # Recursively summarize subdirectory
            summarize_dir "$item" $((depth+1)) "$max_depth" "$max_items" "$content_ratio" "$max_lines"
        elif [ -f "$item" ]; then
            echo "  $item"
            display_file_content "$item" "$content_ratio" "$max_lines"
        fi
        displayed_count=$((displayed_count + 1))
    done
}

# Function to display trimmed content of a file
display_file_content() {
    local file="$1"
    local content_ratio="$2"
    local max_lines="$3"

    # Determine number of lines in the file
    local total_lines=$(wc -l < "$file")

    # Calculate number of lines to display based on the content ratio
    local lines_to_display=$((total_lines * content_ratio / 100))
    
    # Ensure lines_to_display is within the max_lines limit
    if [ "$lines_to_display" -gt "$max_lines" ]; then
        lines_to_display="$max_lines"
    fi

    # Display the first few lines of the file
    echo "Displaying first $lines_to_display lines from $file:"
    head -n "$lines_to_display" "$file"
}

# Default configuration for depth, items, and content ratio
max_depth=3
max_items=10
content_ratio=20  # Show 20% of the file content
max_lines=10      # Maximum of 10 lines per file

# Start summarizing from the provided directory
summarize_dir "$1" 1 "$max_depth" "$max_items" "$content_ratio" "$max_lines"

