from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ConcatItemsd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandAdjustContrastd,
    EnsureTyped,
)
from typing import Sequence


def get_train_transforms(patch_size: Sequence[int] = (144, 128, 16)):
    """
    Training transforms for labeled data.
    Expects keys: "t2w", "adc", "hbv", "seg".
    Outputs keys: "image" (3xHxWxD) and "seg".
    """
    return Compose(
        [
            LoadImaged(keys=["t2w", "adc", "hbv", "seg"]),
            EnsureChannelFirstd(keys=["t2w", "adc", "hbv", "seg"]),
            NormalizeIntensityd(keys=["t2w", "adc", "hbv"], nonzero=True, channel_wise=True),
            ConcatItemsd(keys=["t2w", "adc", "hbv"], name="image", dim=0),
            RandSpatialCropd(keys=["image", "seg"], roi_size=patch_size, random_center=True, random_size=False),
            RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "seg"], prob=0.5, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
            RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.7, 1.5)),
            EnsureTyped(keys=["image", "seg"]),
        ]
    )
