#!/bin/bash
#SBATCH --job-name=treemonitoring
#SBATCH --output=nohup.out
#SBATCH --error=error.out
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=60Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=long
#SBATCH -c 8

module add python/3.9
source "$HOME/bio/bin/activate"

python treemonitoring/models/trainer.py --cfg /home/mila/v/venkatesh.ramesh/scratch/Tree-Monitoring/treemonitoring/models/configs/quebectrees_processor.yaml #--debug
#python treemonitoring/models/trainer.py --cfg /home/mila/v/venkatesh.ramesh/scratch/Tree-Monitoring/treemonitoring/models/configs/quebectrees_utae.yaml
#python treemonitoring/models/trainer.py --cfg /home/mila/v/venkatesh.ramesh/scratch/Tree-Monitoring/treemonitoring/models/configs/treemonitoring_deeplabv3resnet50.yaml
#python treemonitoring/models/trainer.py --cfg /home/mila/v/venkatesh.ramesh/scratch/Tree-Monitoring/treemonitoring/models/configs/treemonitoring_dualgcnresnet50.yaml --debug
#python treemonitoring/models/trainer.py --cfg /home/mila/v/venkatesh.ramesh/scratch/Tree-Monitoring/treemonitoring/models/configs/treemonitoring_unetresnet50.yaml --debug
