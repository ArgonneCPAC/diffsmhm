# diffsmhm
differentiable models of the SMHM

# Installation
Git clone the repo and a `pip install -e .` 

# Environment setup
[Here](https://docs.google.com/document/d/1kyGcjXcY61rw1rEr64ggtCyfRv3w_Sp3G5DlWZVl9_I/edit?usp=sharing) are instructions/notes that cover the setup of a diffsmhm-ready conda environment on Argonne's Polaris system. 

# A Demo Problem
To walk through the main features of this repo, let's create a mock problem.
By picking a set of model parameters, we can generate a wprp measurement, and then run optimization and inference routines on that mock measurement.
To keep this demo relatively computationally light, we'll use a mass cut on the Bolshoi catalog to decrease the number of objects.
I ran this on a single Polaris node in an interactive job, expect about half an hour of combined runtime from the optimization and inference steps.

Also, you'll want to grab the `set_affinity_gpu_polaris.sh` script from the Polaris [documentation](https://docs.alcf.anl.gov/polaris/running-jobs/).

## Create "goal" data
First, we generate the "goal" wprp measurement using `scripts/compute_one_wprp.py`:
```
mpirun -np 8 ./set_affinity_gpu_polaris python compute_one_wprp.py --halo-file [PATH_TO_HALOS] --particle-file [PATH_TO_PARTICLES] -p 0.02 --hmcut 14.6
```

This will generate a single wprp measurement with a random set of parameters that vary by +/- 2% from the default set, and use a host mass cut of 14.6 when loading the catalog to greatly decrease the number of objects.
The script will save two files which we'll use in the next steps: `wprp_single.npy` and `rpbins_single.npy`.

## Optimization
Next, we can run an optimization to find a starting point for HMC.
This will use the script `scripts/fit_wprp_all.py` like so
```
mpirun -np 8 ./set_affinity_gpu_polaris python fit_wprp_all.py -w wprp_single.npy -r rpbins_single.npy --halo-file [PATH_TO_HALOS] --particle-file [PATH_TO_PARTICLES] ---hmcut 14.6 --adam-tmax 10 --adam-a 0.005
```

The above will run the Adam optimizer for 10 minutes on the SMHM model starting at the default parameters and with our previously generated wprp as the goal value. 
The script generates output every 100 iterations, printing the iteration number, the error of that iteration, and the percent error for each wprp radial bin.
Additionally, the script will generate two figures, `fig_wprp_all_error.png` and `fig_wprp_all_wprp.png` which can be used to confirm that the optimization did actually work.
We have one output file, `theta_opt.npy`, which we will use as our starting point for the HMC code.

## Inference

Finally, we can run HMC on this problem using the starting position obtained through the Adam optimization script.

Run
```
mpirun -np 8 ./set_affinity_gpu_polaris python hmc_wprp_all.py -w wprp_single.npy -r rpbins_single.npy --halo-file [PATH_TO_HALOS] --particle-file [PATH_TO_PARTICLES] -t `theta_opt.npy` --hmcut 14.6 --hmc_niter 500 --hmc_nwarmup 250
```
For me on a single Polaris node, this script took 15 minutes to run.

The script outputs a csv file of the HMC positions, and creates a corner plot.
You can verify on the corner plot that this pipeline 


# Running Code
## Scaling
I ran scaling tests with `diffsmhm/scripts/scale_bolshoi_copy.py`.
This script allows you to clone the Bolshoi volume to "create" a larger simulation to test on. 

The submission scripts `submit_strong_111.sh` and `submit_strong_444.sh`, located in the same directory, demonstrate strong scaling tests with a single Bolshoi volume and a 4x4x4 cloned volume respectively. Additionally, `submit_weak.sh` performs weak scaling using 1x1x1, 2x2x2, 3x3x3, and 4x4x4 volumes. 

## Optimization
The script `diffsmhm/scripts/fit_watson_all_adam.py` fits the full correlation function of the "Watson" dataset using Adam.

## Inference
The script `diffsmhm/scripts/demo_hmc.py` demonstrates a HMC run of a self fit to the Bolshoi catalog using Numpyro HMC. 
