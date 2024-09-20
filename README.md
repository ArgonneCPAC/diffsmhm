# diffsmhm
differentiable models of the SMHM

# Installation
Git clone the repo and a `pip install -e .` 

# Environment setup
[Here](https://docs.google.com/document/d/1kyGcjXcY61rw1rEr64ggtCyfRv3w_Sp3G5DlWZVl9_I/edit?usp=sharing) are instructions/notes that cover the setup of a diffsmhm-ready conda environment on Argonne's Polaris system. 


# A Demo Problem
For more details on the parameters of the scripts, see the README in the scripts directory.

To walk through the main features of this repo, let's create a mock problem.
By picking a set of model parameters, we can generate a wprp measurement, and then run optimization and inference routines on that mock measurement.
To keep this demo relatively computationally light, we'll use a mass cut on the Bolshoi catalog to decrease the number of objects.
I ran this on a single Polaris node in an interactive job, expect about half an hour of combined runtime from the optimization and inference steps.

Also, you'll want to grab the `set_affinity_gpu_polaris.sh` script from the Polaris [documentation](https://docs.alcf.anl.gov/polaris/running-jobs/).
This assigns each rank a single GPU, which greatly improves runtime in this demo case of running 8 MPI ranks across 4 GPUs.
In general, this code supports multiple GPUs per MPI rank, but for the demo use the affinity script.

The optimization and inference scripts we use here take hdf5 files as input, so if using different
data you'll want to make sure you have wprp, wprp error, and radial bins in that format.
The optimization and inference codes expect 1 hdf5 file with keys "wprp", "rpbins", and "wprp_error".
Outputs saving model parameters and HMC output will each be another hdf5 file; one for the parameters theta and one for the HMC adaptation and chain info.

## Create "goal" data
First, we generate the "goal" wprp measurement using `scripts/compute_one_wprp.py`:
```
mpirun -np 8 ./set_affinity_gpu_polaris.sh python compute_one_wprp.py --halo-file [PATH_TO_HALOS] --particle-file [PATH_TO_PARTICLES] -p 0.02 --hmcut 14.6
```

This will generate a single wprp measurement with a random set of parameters that vary by +/- 2% from the default set, and use a host mass cut of 14.6 when loading the catalog to greatly decrease the number of objects.
The script will save an hdf5 file which we'll use in the next steps: `wprp_single.hdf5`.

## Optimization
Next, we can run an optimization to find a starting point for HMC.
This will use the script `scripts/fit_wprp_all.py` like so
```
mpirun -np 8 ./set_affinity_gpu_polaris.sh python fit_wprp_all.py -w wprp_single.hdf5 --halo-file [PATH_TO_HALOS] --particle-file [PATH_TO_PARTICLES] ---hmcut 14.6 --adam-tmax 10 --adam-a 0.005
```

The above will run the Adam optimizer for 10 minutes on the SMHM model starting at the default parameters and with our previously generated wprp as the goal value. 
I found these parameters to produce a good fit, but you can of course tweak them if you like.
The script generates output every 100 iterations, printing the iteration number, the error of that iteration, and the percent error for each wprp radial bin.
Additionally, the script will generate two figures, `fig_wprp_all_error.png` and `fig_wprp_all_wprp.png` which can be used to confirm that the optimization did actually work.
We have one output file, `theta_opt.hdf5`, which we will use as our starting point for the HMC code.

## Inference

Finally, we can run HMC on this problem using the starting position obtained through the Adam optimization script.

Run
```
mpirun -np 8 ./set_affinity_gpu_polaris.sh python hmc_wprp_all.py -w wprp_single.hdf5 --halo-file [PATH_TO_HALOS] --particle-file [PATH_TO_PARTICLES] -t `theta_opt.hdf5` --hmcut 14.6 --hmc-niter 500 -f -1 --hmc-nwarmup 250 --prior-width 0.01
```
For me on a single Polaris node, this script took just under 15 minutes to run.

The script outputs a csv file of the HMC positions and creates a corner plot, `corner_hmc.py`.

# Running Other Code
## Scaling
I ran scaling tests with `diffsmhm/scripts/scale_bolshoi_copy.py`.
This script allows you to clone the Bolshoi volume to mimic a larger simulation volume to test on. 

The submission scripts `submit_strong_111.sh` and `submit_strong_444.sh`, located in the same directory, demonstrate strong scaling tests with a single Bolshoi volume and a 4x4x4 cloned volume respectively.
Additionally, `submit_weak.sh` performs weak scaling using 1x1x1, 2x2x2, 3x3x3, and 4x4x4 volumes. 
