#!/bin/bash
#SBATCH -t 0-05:00
#SBATCH -A def-dkrass
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=9000M
source /home/eliransc/.virtualenvs/deep_queue/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/Deep_queue/code/train_cpu.py

