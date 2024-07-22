#!/bin/bash
#SBATCH --job-name=treemonitoring
#SBATCH --output=nohup.out
#SBATCH --error=error.out
#SBATCH --ntasks=1
#SBATCH --time=80:00:00
#SBATCH --mem=60Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=long
#SBATCH -c 8

module add python/3.9
source "$HOME/bio/bin/activate"

python treemonitoring/models/trainer.py --cfg /home/mila/v/venkatesh.ramesh/scratch/Tree-Monitoring/treemonitoring/models/configs/quebectrees_unet3d_99.yaml #--debug
