import glob
import os
import nibabel as nib
import numpy as np
from monai.metrics import compute_meandice, compute_hausdorff_distance

# Set directories
inference_dir = "D:\\project_data\\spleen_dataset\\Task09_Spleen_f\\results_train_inference\\spleen_seg_no_augment_epoch_1000"
nifti_dir = 'D:\\project_data\\spleen_dataset\\Task09_Spleen_f\\Task09_Spleen\\imagesTr'
labels_dir = "D:\\project_data\\spleen_dataset\\Task09_Spleen_f\\Task09_Spleen\\labelsTr"

# Get file paths
inference_files = sorted(glob.glob(os.path.join(inference_dir, "*.nii*")))
nifti_files = sorted(glob.glob(os.path.join(nifti_dir, "*.nii*")))
labels_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii*")))

# Iterate over files
for i, (inf_file, nifti_file, labels_file) in enumerate(zip(inference_files, nifti_files, labels_files)):
    # Load images
    inf_image = nib.load(inf_file).get_fdata()
    nifti_image = nib.load(nifti_file).get_fdata()
    labels_image = nib.load(labels_file).get_fdata()

    # Compute metrics
    dice = compute_meandice(inf_image, nifti_image, include_background=True)
    #jaccard = compute_jaccard_index(inf_image, nifti_image, include_background=True)
    hausdorff = compute_hausdorff_distance(inf_image, labels_image)

    # Print results
    print(f"File {i+1}:")
    print(f"Dice coefficient: {dice:.4f}")
    #print(f"Jaccard index: {jaccard:.4f}")
    print(f"Hausdorff distance: {hausdorff:.4f}")
