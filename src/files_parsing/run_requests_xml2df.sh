#!/bin/bash
SBATCH --job-name=requests_xml2df
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00
SBATCH --output=output_requests_xml2df_logs.slurm
SBATCH --error=output_requests_xml2df_errors.slurm
srun -N1 -n1  python run_requests_xml2df.py