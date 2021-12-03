#!/bin/bash
#SBATCH -t 0-23:58
#SBATCH -A def-dkrass
#SBATCH --mem 10000
source /home/eliransc/.virtualenvs/deep_queue/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/Deep_queue/code/sampling_ph_1.py

