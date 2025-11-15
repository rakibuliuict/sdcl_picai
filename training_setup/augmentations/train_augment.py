# from monai.transforms import (
#     Compose,
#     LoadImaged,
#     EnsureChannelFirstd,
#     NormalizeIntensityd,
#     ConcatItemsd,
#     RandSpatialCropd,
#     RandFlipd,
#     RandRotate90d,
#     RandGaussianNoised,
#     RandAdjustContrastd,
#     EnsureTyped,
# )
# from typing import Sequence


# def get_train_transforms(patch_size: Sequence[int] = (144, 128, 16)):
#     """
#     Training transforms for labeled data.
#     Expects keys: "t2w", "adc", "hbv", "seg".
#     Outputs keys: "image" (3xHxWxD) and "seg".
#     """
#     return Compose(
#         [
#             LoadImaged(keys=["t2w", "adc", "hbv", "seg"]),
#             EnsureChannelFirstd(keys=["t2w", "adc", "hbv", "seg"]),
#             NormalizeIntensityd(keys=["t2w", "adc", "hbv"], nonzero=True, channel_wise=True),
#             ConcatItemsd(keys=["t2w", "adc", "hbv"], name="image", dim=0),
#             RandSpatialCropd(keys=["image", "seg"], roi_size=patch_size, random_center=True, random_size=False),
#             RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0),
#             RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=1),
#             RandRotate90d(keys=["image", "seg"], prob=0.5, max_k=3),
#             RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
#             RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.7, 1.5)),
#             EnsureTyped(keys=["image", "seg"]),
#         ]
#     )

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ConcatItemsd,
    Orientationd,
    ResizeWithPadOrCropd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandAdjustContrastd,
    EnsureTyped,
)

def get_train_transforms():
    patch_size = (144, 128, 16)  # target (H, W, D)

    return Compose([
        LoadImaged(keys=["t2w", "adc", "hbv", "seg"]),
        EnsureChannelFirstd(keys=["t2w", "adc", "hbv", "seg"]),

        # Optional but recommended: unify orientation
        Orientationd(keys=["t2w", "adc", "hbv", "seg"], axcodes="RAS"),

        NormalizeIntensityd(keys=["t2w", "adc", "hbv"], nonzero=True, channel_wise=True),

        # stack modalities into a single 3-channel image
        ConcatItemsd(keys=["t2w", "adc", "hbv"], name="image", dim=0),

        #  This is the important part: force SAME spatial size for all
        ResizeWithPadOrCropd(keys=["image", "seg"], spatial_size=patch_size),

        # data augmentation
        RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "seg"], prob=0.5, max_k=3),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
        RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.7, 1.5)),

        EnsureTyped(keys=["image", "seg"]),
    ])
