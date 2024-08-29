#!/bin/bash -l

#PBS -N wprp_fit
#PBS -l select=10:system=polaris
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
export JAX_ENABLE_X64=True

module use /soft/modulefiles
module load conda
conda activate diffsmhm

#module load cudatoolkit-standalone/12.5.0

cd $PBS_O_WORKDIR

# relative from the diffsmhm/scripts dir
WPRP_FILE="inputs_watson/wprp_watson10.6.all.npy"
WPERR_FILE="inputs_watson/wperr_watson10.6.all.npy"
RP_FILE="inputs_watson/rpbins_watson10.6.all.npy"

ADAM_A=0.01
OUTDIR="output/"

mpiexec -np ${NTOTRANKS} --ppn ${NRANKS_PER_NODE}:node --depth=${NDEPTH} --cpu-bind depth \
    ./set_gpu_affinity.sh python fit_wprp_all.py -w $WPRP_FILE -e $WPERR_FILE -r $RP_FILE \
    --adam_a $ADAM_A --o $OUTDIR
