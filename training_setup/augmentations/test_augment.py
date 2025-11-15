# from monai.transforms import (
#     Compose,
#     LoadImaged,
#     EnsureChannelFirstd,
#     NormalizeIntensityd,
#     ConcatItemsd,
#     EnsureTyped,
# )
# from typing import Sequence


# def get_test_transforms(patch_size: Sequence[int] = (144, 128, 16)):
#     """
#     Validation / test transforms.
#     """
#     return Compose(
#         [
#             LoadImaged(keys=["t2w", "adc", "hbv", "seg"]),
#             EnsureChannelFirstd(keys=["t2w", "adc", "hbv", "seg"]),
#             NormalizeIntensityd(keys=["t2w", "adc", "hbv"], nonzero=True, channel_wise=True),
#             ConcatItemsd(keys=["t2w", "adc", "hbv"], name="image", dim=0),
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
    EnsureTyped,
)

def get_test_transforms():
    patch_size = (144, 128, 16)

    return Compose([
        LoadImaged(keys=["t2w", "adc", "hbv", "seg"]),
        EnsureChannelFirstd(keys=["t2w", "adc", "hbv", "seg"]),
        Orientationd(keys=["t2w", "adc", "hbv", "seg"], axcodes="RAS"),
        NormalizeIntensityd(keys=["t2w", "adc", "hbv"], nonzero=True, channel_wise=True),
        ConcatItemsd(keys=["t2w", "adc", "hbv"], name="image", dim=0),

        ResizeWithPadOrCropd(keys=["image", "seg"], spatial_size=patch_size),

        EnsureTyped(keys=["image", "seg"]),
    ])
