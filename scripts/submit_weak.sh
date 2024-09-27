#!/bin/bash -l

#PBS -N wprp_weak
#PBS -l select=320:system=polaris
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

export IBV_FORK_SAFE=1

module use /soft/modulefiles
module load conda
conda activate diffsmhm

cd $PBS_O_WORKDIR

mpirun -np   5 python scale_bolshoi_copy.py --n_copies 1
mpirun -np  40 python scale_bolshoi_copy.py --n_copies 2
mpirun -np 135 python scale_bolshoi_copy.py --n_copies 3
mpirun -np 320 python scale_bolshoi_copy.py --n_copies 4
