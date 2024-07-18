#!/bin/bash
SBATCH --job-name=qrels2df
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00
SBATCH --output=output_qrels2df_logs.slurm
SBATCH --error=output_qrels2df_errors.slurm
srun -N1 -n1  python run_qrels2df.py
