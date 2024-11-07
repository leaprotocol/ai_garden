#!/bin/bash

# Check if at least 5 arguments are provided (directory, filelimit, read limit, depth, enable_second_tree)
if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <directory> <filelimit> <read_limit> <depth> <enable_second_tree>"
  exit 1
fi

# Get the directory, file limit, read limit, depth, and enable_second_tree from the arguments
directory=$1
filelimit=$2    # like 80
read_limit=$3  # Number of characters to randomly read from the file, like 100
depth=$4 # depth, like 3
enable_second_tree=$5 # for debug

# Validate if the directory exists
if [ ! -d "$directory" ]; then
  echo "Error: Directory $directory does not exist."
  exit 1
fi

# Run the first tree command and print its output
echo "Directory structure for $directory (Depth: $depth):"
tree -L "$depth" --filelimit "$filelimit"  -Jshi --sort size --du "$directory"

# Optionally run the second tree command if enabled
if [ "$enable_second_tree" -eq 1 ]; then
    # Now process the files from the tree output
    echo "Processing files..."

    tree -L "$depth" --filelimit "$filelimit" --prune -fai --sort size "$directory" | while read filepath; do
      if [ -f "$filepath" ]; then
        # Get the file path relative to the start directory
        relative_filepath=$(realpath --relative-to="$directory" "$filepath")

        # Get file size
        filesize=$(stat --format="%s" "$filepath")

        # Display file information, file, size, type
        echo "f: $relative_filepath; s: $filesize; t: $(file --mime-type -b "$filepath")" 

        # Get the MIME type and decide whether to process the file
        mimetype=$(file --mime-type -b "$filepath")
        if echo "$mimetype" | grep -qE 'text|application/(javascript|json|xml|x-sh|x-python|x-markdown)'; then
          # If file size is greater than the read limit, pick a random position
          if [ "$filesize" -gt "$read_limit" ]; then
            # Generate a random starting point ensuring we can still read $read_limit chars
            random_offset=$((RANDOM % (filesize - read_limit)))
            echo "Filehead:"
            head -c "$read_limit" "$filepath" | tr '\n' ' '
            echo -e "\nRandomcontent($random_offset):"
           
            # Use dd to extract $read_limit chars from a random position
            dd if="$filepath" bs=1 skip="$random_offset" count="$read_limit" 2>/dev/null | tr '\n' ' '
          else
            # If file is too small, just print the whole file
            echo "Filecontent:"
            cat "$filepath" | tr '\n' ' '
          fi
          echo -e
        fi
      fi
    done
fi

