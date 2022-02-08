#!/bin/bash
#SBATCH -t 0-0:18
#SBATCH -A def-dkrass
source /home/d/dkrass/eliransc/queues/bin/activate
python /scratch/d/dkrass/eliransc/Deep_queue/code/sample_trail.py
