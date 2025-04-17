#!/bin/bash
# Script to run all benchmark jobs in sequence

echo "Starting benchmark jobs..."

# Submit the first job (benchmark_u2net)
echo "Submitting benchmark_u2net.sb..."
job_id1=$(sbatch benchmark_u2net.sb | awk '{print $4}')
echo "Job ID: $job_id1"

# Submit the second job to run after the first one completes successfully
echo "Submitting benchmark_u2netp.sb (depends on $job_id1)..."
job_id2=$(sbatch --dependency=afterok:$job_id1 benchmark_u2netp.sb | awk '{print $4}')
echo "Job ID: $job_id2"

# Submit the third job to run after the second one completes successfully
echo "Submitting scaling_u2net.sb (depends on $job_id2)..."
job_id3=$(sbatch --dependency=afterok:$job_id2 scaling_u2net.sb | awk '{print $4}')
echo "Job ID: $job_id3"

# Submit the fourth job to run after the third one completes successfully
echo "Submitting scaling_u2netp.sb (depends on $job_id3)..."
job_id4=$(sbatch --dependency=afterok:$job_id3 scaling_u2netp.sb | awk '{print $4}')
echo "Job ID: $job_id4"

echo "All jobs submitted successfully!"
echo "Job IDs: $job_id1 -> $job_id2 -> $job_id3 -> $job_id4"
echo "You can monitor job status with: squeue -u $USER" 