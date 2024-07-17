#!/bin/bash
#SBATCH --job-name=AE
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00
#SBATCH --output=output_AE_logs.slurm
#SBATCH --error=output_AE_errors.slurm
srun -N1 -n1  python run_aspects_extraction.py
