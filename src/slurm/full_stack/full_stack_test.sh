#!/bin/bash
#SBATCH --job-name=mpi4py_test
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=4G

set -e
set -v
# set -x

module purge
module load gnu12/12.3.0
module load openmpi4

python3.11 -m venv .vhpc
. .vhpc/bin/activate
venv=$(echo "$VIRTUAL_ENV")
if [ $venv != "" ] ; then
    echo "Virtual environment is activated: $VIRTUAL_ENV"
else
    echo "Failed to activate virtual environment."
    exit 1
fi
pip install --upgrade pip
pip install mpi4py

mpirun -N $SLURM_JOB_NUM_NODES ${venv}/bin/python3.11 -m mpi4py test.py
deactivate
rm -rf .vhpc

exit 0
