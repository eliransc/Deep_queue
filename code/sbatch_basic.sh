#!/bin/bash
#SBATCH -t 0-03:00
#SBATCH -A def-dkrass
source /home/eliransc/.virtualenvs/deep_queue/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/Deep_queue/code/sampling_ph.py

