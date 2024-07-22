import argparse
import os

import segmentation_models_pytorch as smp
import yaml
from processor import ProcessorDeeplabV3Plus, ProcessorUnet
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

from treemonitoring.models.model import Model
from treemonitoring.models.pastis_model_utils import get_model


def train_model(arch: str, n_class: int) -> None:
    try:
        if arch == "deeplabv3resnet50":
            deeplabv3 = deeplabv3_resnet50(
                weights_backbone="IMAGENET1K_V2", progress=True, num_classes=n_class
            )
            Model(deeplabv3).train()
        elif arch == "deeplabv3resnet101":
            deeplabv3 = deeplabv3_resnet101(
                weights_backbone="IMAGENET1K_V2", progress=True, num_classes=n_class
            )
            Model(deeplabv3).train()
        elif arch == "unetresnet50":
            unet = smp.Unet(
                encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=n_class
            )
            Model(unet).train()
        elif arch == "processor_unet":
            model = ProcessorUnet(n_class)
            Model(model).train()
        elif arch == "processor_deeplab":
            model = ProcessorDeeplabV3Plus(n_class)
            Model(model).train()
        else:
            model = get_model(arch, num_classes=n_class, mode="semantic")
            Model(model).train()
    except Exception as e:
        print(e)
        # raise Exception("Architecture {} is not yet supported.".format(arch))


def _load_cfg(cfg_path):
    with open(cfg_path, "r") as stream:
        cfg = yaml.safe_load(stream)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", help="Path to config file.")
parser.add_argument("--ckp", help="Path to checkpoint to load.", default=None)
parser.add_argument("--debug", help="Debug mode to switch wand mode offline", action="store_true")
args = parser.parse_args()
cfg = _load_cfg(args.cfg)

if args.debug:
    os.environ["WANDB_MODE"] = "offline"

train_model(cfg["model"]["name"], cfg["dataset"]["n_classes"])
