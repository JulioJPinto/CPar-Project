#!/bin/sh
#
#SBATCH --exclusive     # exclusive node for the job
#SBATCH --time=05:00    # allocation for 2 minutes
#SBATCH --constraint=k20

time ./fluid_sim
