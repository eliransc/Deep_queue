#!/bin/bash
#SBATCH -t 1-00:00
#SBATCH -A def-dkrass
source /home/d/dkrass/eliransc/queues/bin/activate
python /scratch/d/dkrass/eliransc/redirected_call/code/main_accuracy.py

