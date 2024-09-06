# Detailed Script Documentation

## `compute_one_wprp.py`

This script isn't for analysis purposes, I found that it was useful to have an easy way to compute a single wprp measurement for plotting or testing purposes.

With this script, wprp can be computed either with randomized parameters, or with a specific parameter set.
If no specific parameter set is provided, the script will generate randomized parameters according to the seed and perturbation limit provided (or defaults or none are provided).
If a parameter set is provided, any randomization arguments are ignored.

We have a few command line options:
* `--halo-file`:
    Path to the Bolshoi halo catalog. Defaults to my data directory on Polaris.
* `--particle-file`:
    Path to the Bolshoi particle catalog. Default to my data directory on Polaris.
* `--mass-bin-low`:
    Lower limit mass bin for selection function. Defaults to 10.6
* `--mass-bin-high`:
    Upper limit mass bin for selection function. Defaults to 100.0
* `-r`, `--rpbins`:
    Filepath to `.npy` file that stores radial bins for wprp. Defaults to "watson-like" bins.
* `-t`, `--theta`:
    Model parameters for which to compute wprp. Defaults to model defaults.
* `-p`, `--perturbation-limit`:
    Percentage limit to perturb parameters by when doing a randomized computation. Providing `--theta` will override this argument and wprp will be compute based on `--theta`.
* `-s`, `--perturbation-seed`:
    Numpy randomization seed to use for parameter randomization. Providing `--theta` will override this argument and any provided seed will be ignored.
* `-o`, `--outdir`:
    Filepath prefix for output files. Default is "./"
* `--hmcut`:
    "Host mpeak cut" used when loading data. Higher values result in less data being used and a faster computation, which is useful for testing or demo problems.

Outputs:
* `outdir/wprp_single.npy` : the computed wprp measurement
* `outdir/rpbins_single.npy` : the rpbins used in the wprp measurement

## `fit_wprp_all.py`

Fits the full correlation function to a provided wprp measurement using the Adam optimizer.

Command line options:
* `--halo-file`:
    Path to the Bolshoi halo catalog. Defaults to my data directory on Polaris.
* `--particle-file`:
    Path to the Bolshoi particle catalog. Default to my data directory on Polaris.
* `-w`, `--wprp`:
    Filepath to `.npy` file that stores the "goal" wprp measurement. Required.
* `-e`, `--wprp-error`:
    Filepath to `.npy` file that stores uncertainty in "goal" wprp measurement. Defaults to 10% of "goal" measurement.
* `--mass-bin-low`:
    Lower limit mass bin for selection function. Defaults to 10.6
* `--mass-bin-high`:
    Upper limit mass bin for selection function. Defaults to 100.0
* `-r`, `--rpbins`:
    Filepath to `.npy` file that stores radial bins for wprp. Defaults to "watson-like" bins.
* `-t`, `--theta-init`:
    Filepath t0 `.npy` file that stores initial parameter set. Defaults to model defaults.
* `-o`, `--outdir`:
    Filepath prefix for output files. Default is "./"
* `--hmcut`:
    "Host mpeak cut" used when loading data. Higher values result in less data being used and a faster computation, which is useful for testing or demo problems.
* `--adam-a`:
    Step size parameter for the Adam optimizer. Defaults to 0.001.
* `--adam-b1`:
    b1 parameter for the Adam optimizer; influences step size decay rate. Defaults to 0.9.
* `--adam-b2`:
    b2 parameter for the Adam optimizer; influenecs step size decay rate. Defaults to 0.999.
* `--adam-tmax`:
    Maximum number of minutes to run the Adam optimzier for. Default is 50.
* `--adam-err-thresh`:
    Error threshold at which to stop optimization. Default is 1e-6.
* `-p`, `--print-rate`:
    Rate at which optimization information is printed. Default is every 100 iterations.

Outputs:
* `outdir/theta_opt.npy`: Resulting parameter set after optimization
* `outdir/fig_wprp_all_wprp.png` : Figure showing starting, goal, and final wprp.
* `outdir/fig_wprp_all_error.png` : Figure showing error history of the optimization.

## `hmc_wprp_all.py`

Performs HMC over the full wprp function around a specified prior location.

Command line options:
* `--halo-file`:
    Path to the Bolshoi halo catalog. Defaults to my data directory on Polaris.
* `--particle-file`:
    Path to the Bolshoi particle catalog. Default to my data directory on Polaris.
* `-w`, `--wprp`:
    Filepath to `.npy` file that stores the "goal" wprp measurement. Required.
* `-e`, `--wprp-error`:
    Filepath to `.npy` file that stores uncertainty in "goal" wprp measurement. Defaults to 10% of "goal" measurement.
* `--mass-bin-low`:
    Lower limit mass bin for selection function. Defaults to 10.6
* `--mass-bin-high`:
    Upper limit mass bin for selection function. Defaults to 100.0
* `-r`, `--rpbins`:
    Filepath to `.npy` file that stores radial bins for wprp. Defaults to "watson-like" bins.
* `-t`, `--theta-init`:
    Filepath t0 `.npy` file that stores initial parameter set. Defaults to model defaults.
* `-o`, `--outdir`:
    Filepath prefix for output files. Default is "./"
* `--hmcut`:
    "Host mpeak cut" used when loading data. Higher values result in less data being used and a faster computation, which is useful for testing or demo problems.
* `--prior-width`:
    Standard deviation to apply to the priors as a percent of the allowed interval width. Default is 5%.
* `--hmc-niter`:
    Number of HMC iterations to perform. Default is 1000.
* `--hmc-nwarmup`:
    Number of warmup iterations to perform. Default is 500.

Outputs:
* `outdir/positions.csv`:
    Positions from HMC states.
* `outdir/corne_hmc.png`:
    A corner plot of the HMC locations.

## `scale_bolshoi_copy.py`

Script that times wprp measurements for use in scaling tests.

Command line options:
* `--halo-file`:
    Path to the Bolshoi halo catalog. Defaults to my data directory on Polaris.
* `--hmcut`:
    "Host mpeak cut" used when loading data. Higher values result in less data being used and a faster computation, which is useful for testing or demo problems, or for reducing memory usage when scaling.
* `--n-copies`:
    Number of Bolshoi copies to make in each dimension. Default is 1. Note that we require n-ranks >= n-copies^3 for data loading purposes.
* `--n-iter`:
    Number of wprp computation iterations to perform and average for a final time. Default is 5.
