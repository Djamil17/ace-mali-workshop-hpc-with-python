#!/bin/bash

# Script: sum_numbers.sh
#
# Description:
#   This script calculates the sum of numbers from 1 to 1000 and prints the result.
#
# Usage:
#   ./sum_numbers.sh 1 100 2 out.txt
#
# Notes:
#   This script uses a simple loop to iteratively add numbers from 1 to 1000.
#   It demonstrates basic arithmetic operations and variable usage in Bash scripting.
#
# Example output:
#   sum of 1-1000 is 500500
#

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
counter=0

for (( i=start; i<=end; i+=step )); do
    counter=$(($counter+$i))
    echo -n "${i}" >> ${output_file}.txt
done
echo "sum of ${start}-${end} is $counter"

rm ${output_file}.txt

exit 0
