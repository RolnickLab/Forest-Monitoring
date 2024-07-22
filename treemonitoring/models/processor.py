import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


class Processor(nn.Module):
    def __init__(self):
        super(Processor, self).__init__()
        self.layer1 = nn.Sequential(
            # Config base
            nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(2, 3, 3), padding=(0, 1, 1))
        )

    def forward(self, x):
        output = self.layer1(x)
        return output


class ProcessorUnet(nn.Module):
    def __init__(self, n_class):
        super(ProcessorUnet, self).__init__()
        self.processor = Processor()
        self.unet = smp.Unet(
            encoder_name="resnet50", encoder_weights=None, in_channels=64, classes=n_class
        )

    def forward(self, x, batch_positions=None):
        out1 = self.processor(x)
        out2 = self.unet(out1.squeeze())
        return out2


class ProcessorDeeplabV3Plus(nn.Module):
    def __init__(self, n_class):
        super(ProcessorDeeplabV3Plus, self).__init__()
        self.processor = Processor()
        self.deeplab = smp.DeepLabV3Plus(
            encoder_name="resnet50", encoder_weights=None, in_channels=64, classes=n_class
        )

    def forward(self, x, batch_positions=None):
        out1 = self.processor(x)
        out2 = self.deeplab(out1.squeeze())
        return out2


if __name__ == "__main__":
    model = ProcessorDeeplabV3Plus(16)
    x = torch.rand((2, 3, 4, 768, 768))
    out = model(x)
    print(out.shape)