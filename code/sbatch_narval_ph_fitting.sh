#!/bin/bash
#SBATCH -t 0-23:58
#SBATCH -A def-dkrass
#SBATCH --mem 10000
source /home/eliransc/projects/def-dkrass/eliransc/virenv/queues/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/Deep_queue/code/fitting_ph.py