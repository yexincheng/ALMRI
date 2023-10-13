from monai.apps.deepedit.transforms import (
    AddGuidanceSignalDeepEditd,
    AddRandomGuidanceDeepEditd,
    FindDiscrepancyRegionsDeepEditd,
    NormalizeLabelsInDatasetd,
    FindAllValidSlicesMissingLabelsd,
    AddInitialSeedPointMissingLabelsd,
    SplitPredsLabeld,
)
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
    Resized,
    ScaleIntensityRanged,
    ToNumpyd,
    ToTensord,
    SpatialCropd,
    RandCropByLabelClassesd,
    CenterSpatialCropd,
)


def get_pre_transforms(labels, spatial_size):
    t = [
        LoadImaged(keys=("image", "label"), reader="ITKReader"),
        EnsureChannelFirstd(keys=("image", "label")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # This transform may not work well for MR images
#         ScaleIntensityRanged(keys="image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
        RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
        RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
        RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),

        NormalizeLabelsInDatasetd(keys="label", label_names=labels),
        CenterSpatialCropd(keys=["image", "label"],roi_size=spatial_size),
        #RandCropByLabelClassesd(keys=["image", "label"], label_key="label",spatial_size=[64,64,32],num_classes=4,num_samples=4),
		#SpatialCropd(keys=["image", "label"],roi_center=(144,117,62), roi_size=(256,128,128)),
        #Resized(keys=("image", "label"), spatial_size=spatial_size, mode=("area", "nearest")),
        # Transforms for click simulation
        FindAllValidSlicesMissingLabelsd(keys="label", sids="sids"),	
        AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids"),
        AddGuidanceSignalDeepEditd(keys="image", guidance="guidance"),
        #
        ToTensord(keys=("image", "label")),
    ]

    return Compose(t)

def get_val_pre_transforms(labels, spatial_size):
    t = [
        LoadImaged(keys=("image", "label"), reader="ITKReader"),
        EnsureChannelFirstd(keys=("image", "label")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # This transform may not work well for MR images
#         ScaleIntensityRanged(keys="image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),

        NormalizeLabelsInDatasetd(keys="label", label_names=labels),
        CenterSpatialCropd(keys=["image", "label"],roi_size=spatial_size),
        #RandCropByLabelClassesd(keys=["image", "label"], label_key="label",spatial_size=[64,64,32],num_classes=4,num_samples=4),
		#SpatialCropd(keys=["image", "label"],roi_center=(144,117,62), roi_size=(256,128,128)),
        #Resized(keys=("image", "label"), spatial_size=spatial_size, mode=("area", "nearest")),
        # Transforms for click simulation
        FindAllValidSlicesMissingLabelsd(keys="label", sids="sids"),	
        AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids"),
        AddGuidanceSignalDeepEditd(keys="image", guidance="guidance"),
        #
        ToTensord(keys=("image", "label")),
    ]

    return Compose(t)

def get_click_transforms():
    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),
        ToNumpyd(keys=("image", "label", "pred")),
        # Transforms for click simulation
        FindDiscrepancyRegionsDeepEditd(keys="label", pred="pred", discrepancy="discrepancy"),
        AddRandomGuidanceDeepEditd(
            keys="NA",
            guidance="guidance",
            discrepancy="discrepancy",
            probability="probability",
        ),
        AddGuidanceSignalDeepEditd(keys="image", guidance="guidance"),
        #
        ToTensord(keys=("image", "label")),
    ]

    return Compose(t)


def get_post_transforms(labels):
    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(
            keys=("pred", "label"),
            argmax=(True, False),
            to_onehot=(len(labels), len(labels)),
        ),
        # This transform is to check dice score per segment/label
        SplitPredsLabeld(keys="pred"),
    ]
    return Compose(t)