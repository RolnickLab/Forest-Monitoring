from treemonitoring.models.pastis_models import convgru, convlstm, fpn, unet3d, utae


def get_model(arch, num_classes=16, mode="semantic"):
    if mode == "semantic":
        if arch == "utae":
            model = utae.UTAE(
                input_dim=3,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, num_classes],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                agg_mode="att_group",
                encoder_norm="group",
                n_head=16,
                d_model=256,
                d_k=4,
                encoder=False,
                return_maps=False,
                pad_value=0,
                padding_mode="reflect",
            )
        elif arch == "unet3d":
            model = unet3d.UNet3D(in_channel=3, n_classes=num_classes, pad_value=0)
        elif arch == "fpn":
            model = fpn.FPNConvLSTM(
                input_dim=3,
                num_classes=num_classes,
                inconv=[32, 64],
                n_levels=4,
                n_channels=64,
                hidden_size=88,
                input_shape=(128, 128),
                mid_conv=True,
                pad_value=0,
            )
        elif arch == "convlstm":
            model = convlstm.ConvLSTM_Seg(
                num_classes=num_classes,
                input_size=(128, 128),
                input_dim=10,
                kernel_size=(3, 3),
                hidden_dim=160,
            )
        elif arch == "convgru":
            model = convgru.ConvGRU_Seg(
                num_classes=num_classes,
                input_size=(128, 128),
                input_dim=10,
                kernel_size=(3, 3),
                hidden_dim=180,
            )
        elif arch == "uconvlstm":
            model = utae.RecUNet(
                input_dim=10,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=64,
                encoder=False,
                padding_mode="zeros",
                pad_value=0,
            )
        elif arch == "buconvlstm":
            model = utae.RecUNet(
                input_dim=10,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=30,
                encoder=False,
                padding_mode="zeros",
                pad_value=0,
            )
        return model
    else:
        raise NotImplementedError