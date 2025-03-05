import os
import numpy as np
import pydicom
import cv2 as cv
import nibabel as nib
from pydicom.pixel_data_handlers.util import apply_voi_lut
from glob import glob

dicom_folder = "C:/Users\sarra/Downloads/ORIG_3D_FSPGR_20_Average/ORIG_3D_FSPGR_20_Average"  

dicom_files = sorted(glob(os.path.join(dicom_folder, "*.dcm")))

slices = [pydicom.dcmread(f) for f in dicom_files]
slices.sort(key=lambda x: x.InstanceNumber) 

image_stack = np.stack([apply_voi_lut(s.pixel_array, s) for s in slices], axis=0)

image_stack = (image_stack - np.min(image_stack)) / (np.max(image_stack) - np.min(image_stack)) * 255
image_stack = image_stack.astype(np.uint8)

threshold_value = 127
max_value = 255

threshold_types = {
    "binary": cv.THRESH_BINARY,
    "binary_inv": cv.THRESH_BINARY_INV,
    "trunc": cv.THRESH_TRUNC,
    "tozero": cv.THRESH_TOZERO,
    "tozero_inv": cv.THRESH_TOZERO_INV
}

for name, method in threshold_types.items():
    processed_stack = np.zeros_like(image_stack)

    for i in range(image_stack.shape[0]): 
        _, processed_stack[i] = cv.threshold(image_stack[i], threshold_value, max_value, method)

    nii_image = nib.Nifti1Image(processed_stack, affine=np.eye(4))
    nib.save(nii_image, f"placenta_{name}.nii")

    print(f"Processing complete. Saved as 'placenta_{name}.nii'")

print("All thresholding methods applied and saved as NIfTI files.")