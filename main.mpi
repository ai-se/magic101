#!/bin/bash

#SBATCH -J OIL  # job name
#SBATCH -o out/job.%j.out # name of stdout
#SBATCH -e err/job.%j.err # name of stderr
#SBATCH -N 1   # total number of nodes
#SBATCH -n 10   # total number of mpi cores
#SBATCH -x c[79-98,101-107]  # avoid
#SBATCH -t 21:00:00   # rum time (hhmmss)

cd /home/txia4/magic101
source addroot.sh
/home/txia4/anaconda3/bin/python3 Main/experiment.py $1 $2 10
