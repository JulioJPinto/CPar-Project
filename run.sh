#!/bin/sh
#
#SBATCH --exclusive     # exclusive node for the job
#SBATCH --time=05:00    # allocation for 2 minutes

export OMP_NUM_THREADS=20
time ./fluid_sim
