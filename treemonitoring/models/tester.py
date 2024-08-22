import argparse
import os
import segmentation_models_pytorch as smp
import yaml
from processor import ProcessorDeeplabV3Plus, ProcessorUnet
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from treemonitoring.models.model import Model
from treemonitoring.models.pastis_model_utils import get_model

def test_model(arch: str, n_class: int, ckpt_path: str, save_path: str) -> None:
    try:
        if arch == "processor_unet":
            model = ProcessorUnet(n_class)
            Model(model).get_predictions(ckpt_path, save_path)
        elif arch == "processor_deeplab":
            model = ProcessorDeeplabV3Plus(n_class)
            Model(model).get_predictions(ckpt_path, save_path)
        else:
            model = get_model(arch, num_classes=n_class, mode="semantic")
            Model(model).get_predictions(ckpt_path, save_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        # raise Exception(f"Architecture {arch} is not yet supported.")

def load_cfg(cfg_path):
    with open(cfg_path, "r") as stream:
        cfg = yaml.safe_load(stream)
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="Path to config file.")
    parser.add_argument("--ckp", help="Path to checkpoint to load.", default=None)
    parser.add_argument("--savedir", help="Path to save output images.", default=None)
    parser.add_argument("--debug", help="Debug mode to switch wand mode offline", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    if args.debug:
        os.environ["WANDB_MODE"] = "offline"

    ckpt_path = args.ckp
    save_path = str(args.savedir)
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    test_model(cfg["model"]["name"], cfg["dataset"]["n_classes"], ckpt_path, save_path)

if __name__ == "__main__":
    main()