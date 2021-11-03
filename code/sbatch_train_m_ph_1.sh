#!/bin/bash
#SBATCH -t 0-03:00
#SBATCH -A def-dkrass
#SBATCH --gpus-per-node=1
source /home/eliransc/.virtualenvs/deep_queue/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/Deep_queue/code/train_m_ph_1.py

