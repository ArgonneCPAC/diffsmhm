#!/bin/bash -l

#PBS -N wprp_strong_444
#PBS -l select=150:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:10:00
#PBS -l filesystems=home
#PBS -q prod
#PBS -A darkskyml_aesp

NNODES=`wc -l < $PBS_NODEFILE`

NRANKS_PER_NODE=4
NDEPTH=8
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
#NTOTRANKS=8

export IBV_FORK_SAFE=1

module use /soft/modulefiles
module load conda
conda activate blanktest

cd branches_diffsmhm/opt_wprp/diffsmhm/scripts

mpirun -np 1200 ./set_gpu_affinity.sh python scale_bolshoi_copy.py --n_copies 4
mpirun -np 1100 ./set_gpu_affinity.sh python scale_bolshoi_copy.py --n_copies 4
mpirun -np 1000 ./set_gpu_affinity.sh python scale_bolshoi_copy.py --n_copies 4
mpirun -np  900 ./set_gpu_affinity.sh python scale_bolshoi_copy.py --n_copies 4
mpirun -np  800 ./set_gpu_affinity.sh python scale_bolshoi_copy.py --n_copies 4
mpirun -np  700 ./set_gpu_affinity.sh python scale_bolshoi_copy.py --n_copies 4
mpirun -np  600 ./set_gpu_affinity.sh python scale_bolshoi_copy.py --n_copies 4
mpirun -np  500 ./set_gpu_affinity.sh python scale_bolshoi_copy.py --n_copies 4
mpirun -np  400 ./set_gpu_affinity.sh python scale_bolshoi_copy.py --n_copies 4
mpirun -np  300 ./set_gpu_affinity.sh python scale_bolshoi_copy.py --n_copies 4
mpirun -np  200 ./set_gpu_affinity.sh python scale_bolshoi_copy.py --n_copies 4
mpirun -np  100 ./set_gpu_affinity.sh python scale_bolshoi_copy.py --n_copies 4
