#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p subset_output

# Initialize total time variable
total_time=0

# Loop through each file in the subset/ folder
for input_file in subset/*; do
    # Extract the filename from the input file path
    filename=$(basename "$input_file")
    
    # Create a temporary output filename by appending _rm before the file extension
    extension="${filename##*.}"
    name="${filename%.*}"
    temp_output="${name}_rm.${extension}"
    
    start_time=$(date +%s.%N)
    
    rembg i "$input_file" "$temp_output"
    
    end_time=$(date +%s.%N)
    
    elapsed=$(echo "$end_time - $start_time" | bc)
    total_time=$(echo "$total_time + $elapsed" | bc)
    
    cp "$temp_output" subset_output/
    
    # remove the temporary file so it doesn't remain in the current directory.
    # rm "$temp_output"
done

echo "Total rembg processing time (excluding copying): $total_time seconds"
