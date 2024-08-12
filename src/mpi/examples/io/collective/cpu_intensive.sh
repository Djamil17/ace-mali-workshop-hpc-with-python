#!/bin/bash

# Example script that processes a large file
# Usage: ./process_file.sh input_file.txt

input_file="$1"
output_file="processed_output.txt"

# Check if input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' not found."
    exit 1
fi

# Simulate a memory-intensive operation (reading large file)
echo "Processing $input_file..."
# Use 'cat' to read the file into memory
cat "$input_file" > /dev/null

# Perform some CPU-intensive operations (e.g., sorting)
echo "Sorting data..."
sort "$input_file" > "$output_file"

echo "Processing complete. Output saved to $output_file."
