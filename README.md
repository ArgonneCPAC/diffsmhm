# diffsmhm
differentiable models of the SMHM

# Installation
Git clone the repo and a `pip install -e .` 

# Environment setup
[Here](https://docs.google.com/document/d/1kyGcjXcY61rw1rEr64ggtCyfRv3w_Sp3G5DlWZVl9_I/edit?usp=sharing) are instructions/notes that cover the setup of a diffsmhm-ready conda environment on Argonne's Polaris system. 

# Running Code
## Scaling
I ran scaling tests with `diffsmhm/scripts/scale_bolshoi_copy.py`. This script allows you to clone the Bolshoi volume to "create" a larger simulation to test on. 

The submission scripts `submit_strong_111.sh` and `submit_strong_444.sh`, located in the same directory, demonstrate strong scaling tests with a single Bolshoi volume and a 4x4x4 cloned volume respectively. Additionally, `submit_weak.sh` performs weak scaling using 1x1x1, 2x2x2, 3x3x3, and 4x4x4 volumes. 

## Optimization
The script `diffsmhm/scripts/fit_watson_all_adam.py` fits the full correlation function of the "Watson" dataset using Adam.

## Inference
The script `diffsmhm/scripts/demo_hmc.py` demonstrates a HMC run of a self fit to the Bolshoi catalog using Numpyro HMC. 
