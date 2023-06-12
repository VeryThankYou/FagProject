#!/bin/sh
#BSUB -J Fagproject_FeatureExtractionCUDA
#BSUB -o Fagproject_FeatureExtractionCUDA_%J.out
#BSUB -e Fagproject_FeatureExtractionCUDA_%J.err
#BSUB -q gpua100
#BSUB -n 10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=75G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
#BSUB -u erikbuur@hotmail.com
#BSUB -B
#BSUB -N
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
module load python3/3.11.3

# activate the virtual environment 
# NOTE: needs to have been built with the same SciPy version above!
source feature/bin/activate


python3 HFFeatureExtractorCUDA.py
