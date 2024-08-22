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

python treemonitoring/models/trainer.py --cfg treemonitoring/models/configs/processor_unet_7.yaml #--debug
#python treemonitoring/models/tester.py --cfg treemonitoring/models/configs/processor_unet_7.yaml --ckp PATH_TO_CHECKPOINT --savepath OUTPUT_DIR --debug
