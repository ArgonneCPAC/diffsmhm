#!/bin/bash -l

#PBS -N wprp_strong_111
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
conda activate mpi4pyonly_old

# module load cudatoolkit-standalone/12.5.0

cd /home/jwick/branches_diffsmhm/opt_wprp/diffsmhm/scripts

mpirun -np 1200 ./set_gpu_affinity.sh python scale_bolshoi_copy.py 1
mpirun -np 1100 ./set_gpu_affinity.sh python scale_bolshoi_copy.py 1
mpirun -np 1000 ./set_gpu_affinity.sh python scale_bolshoi_copy.py 1
mpirun -np  900 ./set_gpu_affinity.sh python scale_bolshoi_copy.py 1
mpirun -np  800 ./set_gpu_affinity.sh python scale_bolshoi_copy.py 1
mpirun -np  700 ./set_gpu_affinity.sh python scale_bolshoi_copy.py 1
mpirun -np  600 ./set_gpu_affinity.sh python scale_bolshoi_copy.py 1
mpirun -np  500 ./set_gpu_affinity.sh python scale_bolshoi_copy.py 1
mpirun -np  400 ./set_gpu_affinity.sh python scale_bolshoi_copy.py 1
mpirun -np  300 ./set_gpu_affinity.sh python scale_bolshoi_copy.py 1
mpirun -np  200 ./set_gpu_affinity.sh python scale_bolshoi_copy.py 1
mpirun -np  100 ./set_gpu_affinity.sh python scale_bolshoi_copy.py 1

