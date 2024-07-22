#!/bin/bash
#SBATCH --job-name=treemonitoring
#SBATCH --output=nohup.out
#SBATCH --error=error.out
#SBATCH --ntasks=1
#SBATCH --time=80:00:00
#SBATCH --mem=80Gb
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --partition=long
#SBATCH -c 8

module add python/3.9
source $HOME/bio/bin/activate

accelerate launch treemonitoring/models/trainer.py --cfg /home/mila/v/venkatesh.ramesh/scratch/Tree-Monitoring/treemonitoring/models/configs/pastis_utae.yaml
accelerate launch treemonitoring/models/trainer.py --cfg /home/mila/v/venkatesh.ramesh/scratch/Tree-Monitoring/treemonitoring/models/configs/pastis_unet3d.yaml
# accelerate launch treemonitoring/models/trainer.py --cfg /home/mila/v/venkatesh.ramesh/scratch/Tree-Monitoring/treemonitoring/models/configs/pastis_fpn.yaml
# accelerate launch treemonitoring/models/trainer.py --cfg /home/mila/v/venkatesh.ramesh/scratch/Tree-Monitoring/treemonitoring/models/configs/pastis_convlstm.yaml
#accelerate launch treemonitoring/models/trainer.py --cfg /home/mila/v/venkatesh.ramesh/scratch/Tree-Monitoring/treemonitoring/models/treemonitoring_deeplabv3resnet50.yaml #--debug
#accelerate launch treemonitoring/models/trainer.py --cfg /home/mila/v/venkatesh.ramesh/scratch/Tree-Monitoring/treemonitoring/models/treemonitoring_unetresnet50.yaml #--debug
#accelerate launch treemonitoring/models/trainer.py --cfg /home/mila/v/venkatesh.ramesh/scratch/Tree-Monitoring/treemonitoring/models/treemonitoring_dualgcnresnet50.yaml
