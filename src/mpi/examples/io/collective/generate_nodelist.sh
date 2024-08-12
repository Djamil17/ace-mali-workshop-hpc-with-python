#!/bin/bash

################################################################################
# Script: generate_comma_delimited_list.sh
#
# Description:
#   This script generates a comma-delimited list of strings based on input
#   parameters (start, end, step) and writes it to a specified output file.
#   Each element in the list is formatted as "c-node<i>", where <i> is a
#   sequential number within the specified range.
#
# Usage:
#   ./generate_comma_delimited_list.sh <start> <end> <step> <output_file>
#
#   Parameters:
#     <start>: Starting number of the sequence.
#     <end>: Ending number of the sequence.
#     <step>: Step size for incrementing the sequence.
#     <output_file>: File where the comma-delimited list will be saved.
#
# Example:
#   To generate a list from 1 to 10 with a step of 2 and save it to "nodes.txt":
#   ./generate_comma_delimited_list.sh 1 10 2 nodes.txt
#
# Author: Djamil-Lakhdar-Hamina
#
# Date: 06/05/2024
#
################################################################################

set -e

# Check if all required arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <start> <end> <step> <output_file>"
    exit 1
fi

# Assign command-line arguments to variables
start=$1
end=$2
step=$3
output_file=$4

# Initialize an empty string to store the comma-separated list
output=""

# Check if all required arguments are provided
# Loop through the sequence from start to end with the specified step
for (( i=start; i<=end; i+=step )); do
    # Append "c-node${i}" followed by a comma (except after the last element)
    if [ -n "$output" ]; then
        output+=","  # Add comma only if output is not empty
    fi
    output+="c-node${i}"
done

# Write the output string to the specified file
echo -n "$output" > "$output_file"

echo "Comma-delimited list written to $output_file"

exit 0
