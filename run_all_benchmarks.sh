#!/bin/bash --login
# Script to run all benchmark jobs

echo "Starting benchmark jobs..."

# Submit all jobs without dependencies
echo "Submitting benchmark_u2net.sb..."
sbatch benchmark_u2net.sb

echo "Submitting benchmark_u2netp.sb..."
sbatch benchmark_u2netp.sb

echo "Submitting scaling_u2net.sb..."
sbatch scaling_u2net.sb

echo "Submitting scaling_u2netp.sb..."
sbatch scaling_u2netp.sb

echo "All jobs submitted successfully!"
echo "You can monitor job status with: squeue -u $USER" 