#!/bin/bash
#SBATCH --job-name=sbatch-hello-world
#SBATCH --nodes=4
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=2
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# This script submits a Slurm job that runs across 4 nodes with a total of 8 tasks.
# Each node runs 2 tasks. The job prints "Hello world!" along with the job name.
# Output and error messages are saved in files named after the job name and job ID.

working_dir=$(pwd)
job_id=$SLURM_JOB_ID
job_name=$SLURM_JOB_NAME

if [ -d ${working_dir}/tmp ]; then
    echo "directory exists"
else
    echo "creating directory..."
    mkdir ${working_dir}/tmp
fi

srun echo "Hello world! From ${job_name}"

mv *.out $(pwd)/tmp/ &&  mv *.err $(pwd)/tmp/

exit 0
