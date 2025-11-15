import torch
from picai_dataset import prepare_semi_supervised, get_sdcl_dataloaders

def main():
    root = "E:\Dataset\PICAI\ssl_final_data\picai_144_128_16"   # <<< CHANGE THIS

    print("\n===== Testing prepare_semi_supervised() =====")
    lab_loader, unlab_loader, val_loader = prepare_semi_supervised(
        root,
        batch_size_labeled=1,
        batch_size_unlabeled=1,
        batch_size_val=1,
    )

    print(f"\nLabeled train cases: {len(lab_loader.dataset)}")
    print(f"Validation cases:    {len(val_loader.dataset)}")
    print(f"Unlabeled cases:     {len(unlab_loader.dataset)}")

    # ----- Check a labeled sample -----
    print("\n===== Checking ONE LABELED SAMPLE =====")
    img, lab = next(iter(lab_loader))
    print(f"Image shape: {tuple(img.shape)}")   # expect [1, 3, 144, 128, 16]
    print(f"Label shape: {tuple(lab.shape)}")   # expect [1, 1, 144, 128, 16] or [1, 144, 128, 16]

    # ----- Check an unlabeled sample -----
    print("\n===== Checking ONE UNLABELED SAMPLE =====")
    unlab_item = next(iter(unlab_loader))
    weak = unlab_item["weak"]
    strong = unlab_item["strong"]

    print(f"Weak image shape:   {tuple(weak['image'].shape)}")
    print(f"Strong image shape: {tuple(strong['image'].shape)}")

    # ----- Check SDCL dual loaders -----
    print("\n===== Testing get_sdcl_dataloaders() =====")
    lab_a, lab_b, unlab_a, unlab_b, val_loader_sd = get_sdcl_dataloaders(
        root, batch_size=1
    )

    print(f"lab_a len: {len(lab_a.dataset)}, lab_b len: {len(lab_b.dataset)}")
    print(f"unlab_a len: {len(unlab_a.dataset)}, unlab_b len: {len(unlab_b.dataset)}")

    imgA, labA = next(iter(lab_a))
    imgU, _ = next(iter(unlab_a))

    print(f"\nSDCL labeled image shape:   {tuple(imgA.shape)}")
    print(f"SDCL labeled label shape:   {tuple(labA.shape)}")
    print(f"SDCL unlabeled image shape: {tuple(imgU.shape)}")


if __name__ == "__main__":
    main()
