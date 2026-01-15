import os
import pandas as pd
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    ScaleIntensityRangePercentilesd, ToTensord,
    SpatialPadd, RandSpatialCropd, Resized, DivisiblePadd
)


class ColteaPairedDataset(Dataset):
    def __init__(self, csv_file, col_name, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.col_name = col_name
        self.root_dir = root_dir
        self.transform = transform
        self.valid_samples = self._filter_valid_files()

    def _filter_valid_files(self):
        valid_data = []
        patient_ids = self.df[self.col_name].astype(str).tolist()
        for pid in patient_ids:
            art_path = os.path.join(self.root_dir, pid, "arterial.nii.gz")
            nat_path = os.path.join(self.root_dir, pid, "native.nii.gz")
            if os.path.exists(art_path) and os.path.exists(nat_path):
                valid_data.append({"image": art_path, "label": nat_path})
        return valid_data

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        data = self.valid_samples[idx]
        if self.transform:
            data = self.transform(data)
        return data

def get_transforms(mode="train"):
    # Target Input Shape for the Model
    # 256x256 is the standard "high-res" for 3D Generative AI. 
    # 512x512 is too heavy for 3D Attention mechanisms.
    TARGET_XY = 256 
    TARGET_DEPTH = 64
    
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # 1. Normalize Intensity
        ScaleIntensityRangePercentilesd(
            keys=["image", "label"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        ),
        
        # 2. Resize ONLY X and Y (Preserve Original Depth)
        # CHANGE: spatial_size=(256, 256, -1). 
        # "-1" tells MONAI to keep the original Z-dimension (e.g., 150 slices).
        # This prevents the "pancake" distortion.
        Resized(
            keys=["image", "label"], 
            spatial_size=(TARGET_XY, TARGET_XY, -1),
            mode=["trilinear", "trilinear"] 
        ),

        # 3. Pad if smaller than 64 slices
        # If the patient has 30 slices, we pad to 64 so the code doesn't crash.
        # If the patient has 150 slices, this line does nothing (which is what we want).
        SpatialPadd(
            keys=["image", "label"], 
            spatial_size=(TARGET_XY, TARGET_XY, TARGET_DEPTH),
            method="symmetric" 
        ),
    ]

    if mode == "train":
        # 4. Random Crop (The "Patch-Based" Strategy)
        # This extracts a sharp 64-slice chunk from the full volume (e.g., 150 slices).
        # The model sees real pixel resolution, not interpolated mush.
        transforms.append(
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=(TARGET_XY, TARGET_XY, TARGET_DEPTH),
                random_center=True,
                random_size=False
            )
        )
    else:
        # 3. Validation Safety: Divisible Padding
        # This ensures that even if Resize behaves oddly or Z-dim is slightly off, 
        # we pad it to be divisible by 32 (essential for UNet structure).
        transforms.append(
            DivisiblePadd(
                keys=["image", "label"], 
                k=32,
                method="symmetric"
            )
        )
    
    transforms.append(ToTensord(keys=["image", "label"]))
    return Compose(transforms)