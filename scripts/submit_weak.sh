#!/bin/bash -l

#PBS -N wprp_weak
#PBS -l select=300:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
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
conda activate diffsmhm

cd branches_diffsmhm/opt_wprp/diffsmhm/scripts

mpirun -np   28 ./set_gpu_affinity.sh python scale_bolshoi_copy.py 1
mpirun -np  224 ./set_gpu_affinity.sh python scale_bolshoi_copy.py 2
mpirun -np  756 ./set_gpu_affinity.sh python scale_bolshoi_copy.py 3
mpirun -np 1792 ./set_gpu_affinity.sh python scale_bolshoi_copy.py 4
