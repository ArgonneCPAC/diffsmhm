#!/bin/bash -l

#PBS -N wprp_hmc
#PBS -l select=50:system=polaris
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

cd $PBS_O_WORKDIR

# relative from the diffsmhm/scripts dir
WPRP_FILE="inputs_watson/bin10.2/wprp_10.2.hdf5"
MASS_BIN_LOW="10.2"

OUTDIR="outputs_watson/bin10.2/"
THETA_FILE="outputs_watson/bin10.2/theta_opt.hdf5"

PRIOR_WIDTH=0.2

N_WU_ITER=500
N_SAMPLE_ITER=1000

mpiexec -np ${NTOTRANKS} --ppn ${NRANKS_PER_NODE}:node --depth=${NDEPTH} --cpu-bind depth \
    python hmc_wprp_all.py -w $WPRP_FILE --mass-bin-low $MASS_BIN_LOW \
    -o $OUTDIR -t $THETA_FILE -f 100 --hmc-niter $N_SAMPLE_ITER --hmc-nwarmup $N_WU_ITER \
    --prior-width $PRIOR_WIDTH
