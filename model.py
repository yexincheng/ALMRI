from monai.networks.nets import DynUNet

def get_network(network, labels, spatial_size):
    # Network

    network = DynUNet(
        spatial_dims=3,
        in_channels=len(labels) + 1,
        out_channels=len(labels),
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
        norm_name="instance",
        deep_supervision=False,
        res_block=True,
    )
    return network
