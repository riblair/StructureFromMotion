#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --mem=32g
#SBATCH -J "Im Creation"
#SBATCH -A rbe549
#SBATCH -p academic
#SBATCH -t 23:59:59
#SBATCH --gres=gpu:1
#SBATCH --error=SLURM_OUTPUT/slurm_test_%A.err
#SBATCH --output=SLURM_OUTPUT/slurm_test_%A.out
#SBATCH --mail-user=rpblair@wpi.edu
#SBATCH --mail-type=ALL

module load py-pip/24.0 

source ../panovenv/bin/activate

pip install -r requirements.txt

python -u Phase2/Wrapper.py --mode test --scale_factor 8
