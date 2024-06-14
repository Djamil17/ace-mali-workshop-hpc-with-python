#!/bin/bash
#SBATCH --job-name=mpi4py_test
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

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
pip install numpy
pip install numba

mpirun -N $SLURM_JOB_NUM_NODES ${venv}/bin/python3.11 -m mpi4py /Users/dlakhdar/rdct/repos/ace-mali-workshop-hpc-with-python/src/slurm/full_stack/jitted_numpy_monte_carlo.py 10000000
deactivate
rm -rf .vhpc

exit 0
