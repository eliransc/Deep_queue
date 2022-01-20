#!/bin/bash
#SBATCH -t 0-18:58
#SBATCH -A def-dkrass
source /home/d/dkrass/eliransc/queues/bin/activate
python /home/d/dkrass/eliransc/Deep_queue/code/fitting_ph.py
