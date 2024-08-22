#!/bin/bash

python treemonitoring/models/trainer.py --cfg treemonitoring/models/configs/processor_unet_7.yaml #--debug
#python treemonitoring/models/tester.py --cfg treemonitoring/models/configs/processor_unet_7.yaml --ckp PATH_TO_CHECKPOINT --savepath OUTPUT_DIR --debug
