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


def get_unlabeled_weak_transforms(patch_size: Sequence[int] = (144, 128, 16)):
    """
    Weak augmentation for unlabeled data.
    """
    return Compose(
        [
            LoadImaged(keys=["t2w", "adc", "hbv"]),
            EnsureChannelFirstd(keys=["t2w", "adc", "hbv"]),
            NormalizeIntensityd(keys=["t2w", "adc", "hbv"], nonzero=True, channel_wise=True),
            ConcatItemsd(keys=["t2w", "adc", "hbv"], name="image", dim=0),
            RandSpatialCropd(keys=["image"], roi_size=patch_size, random_center=True, random_size=False),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image"], prob=0.5, max_k=3),
            EnsureTyped(keys=["image"]),
        ]
    )


def get_unlabeled_strong_transforms(patch_size: Sequence[int] = (144, 128, 16)):
    """
    Strong augmentation for unlabeled data.
    """
    return Compose(
        [
            LoadImaged(keys=["t2w", "adc", "hbv"]),
            EnsureChannelFirstd(keys=["t2w", "adc", "hbv"]),
            NormalizeIntensityd(keys=["t2w", "adc", "hbv"], nonzero=True, channel_wise=True),
            ConcatItemsd(keys=["t2w", "adc", "hbv"], name="image", dim=0),
            RandSpatialCropd(keys=["image"], roi_size=patch_size, random_center=True, random_size=False),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image"], prob=0.5, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.02),
            RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 1.8)),
            EnsureTyped(keys=["image"]),
        ]
    )
