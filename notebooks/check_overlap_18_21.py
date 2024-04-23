# %%
import numpy as np
from pathlib import Path
import nibabel as nib
from typing import List
import concurrent.futures

# %%
brats2021_label_path: Path = Path(
    "/home/stud/strasser/data/nnUNet_raw_data/Dataset137_BraTS2021/imagesTr"
)
brats2018_label_path: Path = Path(
    "/home/stud/strasser/data/nnUNet_raw_data/Dataset042_BraTS2018/imagesTr"
)

# validation data holds no segmentation labels
# brats2018_validation_path: Path = Path('/home/stud/strasser/archive/brats2018/MICCAI_BraTS_2018_Data_Validation')

# %%
brats2021_label_nii_gz_file_paths: List[Path] = list(brats2021_label_path.glob("*"))
brats2021_label_nii_gz_file_paths[:10]

# %%
brats2018_label_nii_gz_file_paths: List[Path] = list(brats2018_label_path.glob("*"))
brats2018_label_nii_gz_file_paths[:10]

# %%
# # Naive without multitasking

# for brats2018_fp in tqdm(brats2018_segmentation_mask_file_paths):
#     brats2018_segmentation_nparr = nib.load(brats2018_fp).get_fdata()

#     for brats2021_nii_gz_path in brats2021_label_nii_gz_file_paths:
#         brats2021_segmentation_image = nib.load(brats2021_nii_gz_path)
#         brats2021_segmentation_np = brats2021_segmentation_image.get_fdata()

#         # check for equality
#         if np.array_equal(brats2018_segmentation_nparr, brats2021_segmentation_np):
#             raise ValueError(f"Duplicate in segmentation mask for file {brats2021_nii_gz_path}")

# print("no overlap")

# %%
print("brats 2018: ", len(brats2018_label_nii_gz_file_paths))
print("brats 2021: ", len(brats2021_label_nii_gz_file_paths))

print(
    "product: ",
    len(brats2018_label_nii_gz_file_paths)
    * len(brats2021_label_nii_gz_file_paths),
)

# %%
def check_segmentation_equality(brats2018_fp, brats2021_nii_gz_path):
    brats2018_segmentation_nparr = nib.load(brats2018_fp).get_fdata()
    brats2021_segmentation_nparr = nib.load(brats2021_nii_gz_path).get_fdata()

    if np.array_equal(brats2018_segmentation_nparr, brats2021_segmentation_nparr):
        print(f"found duplicate between 2018 {brats2018_fp} and 2021 {brats2021_nii_gz_path}")
        return 1
    return 0

NUM_WORKERS = 16

with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = []
    total = 0

    for brats2018_fp in brats2018_label_nii_gz_file_paths:
        for brats2021_nii_gz_path in brats2021_label_nii_gz_file_paths:
            total += 1
            futures.append(
                executor.submit(
                    check_segmentation_equality, brats2018_fp, brats2021_nii_gz_path
                )
            )

    done = dup = 0
    for future in concurrent.futures.as_completed(futures):
        dup += future.result()
        done += 1
        print(f"checked {done} / {total}\t duplicates: {dup}", end="\r")

print("no overlap")


