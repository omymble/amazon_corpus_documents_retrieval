#!/bin/bash
#SBATCH --job-name=ABSAConversion
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=7-00:00:00
#SBATCH --output=stdout_absa_conversion.slurm
#SBATCH --error=stderr_absa_conversion.slurm

# Load necessary modules or activate your virtual environment if needed
# module load python/3.x
# source /path/to/your/venv/bin/activate

# Run your Python script with the required argument
srun -N1 -n1 python run_absa_xml_to_df.py bert-base-cased
