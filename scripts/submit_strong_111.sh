#!/bin/bash -l

#PBS -N wprp_strong_111
#PBS -l select=300:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -l filesystems=home
#PBS -q prod
#PBS -A darkskyml_aesp

NNODES=`wc -l < $PBS_NODEFILE`

NRANKS_PER_NODE=1
NDEPTH=16
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

export IBV_FORK_SAFE=1

module use /soft/modulefiles
module load conda
conda activate mpi4pyonly_old

cd $PBS_O_WORKDIR

mpirun -np 300 python scale_bolshoi_copy.py
mpirun -np 275 python scale_bolshoi_copy.py
mpirun -np 250 python scale_bolshoi_copy.py
mpirun -np 225 python scale_bolshoi_copy.py
mpirun -np 200 python scale_bolshoi_copy.py
mpirun -np 175 python scale_bolshoi_copy.py
mpirun -np 150 python scale_bolshoi_copy.py
mpirun -np 125 python scale_bolshoi_copy.py
mpirun -np 100 python scale_bolshoi_copy.py
mpirun -np  75 python scale_bolshoi_copy.py
mpirun -np  50 python scale_bolshoi_copy.py
mpirun -np  25 python scale_bolshoi_copy.py

