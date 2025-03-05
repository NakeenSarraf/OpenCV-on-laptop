import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def dice_coefficient(mask1, mask2):
    """
    Compute Dice Similarity Coefficient (DSC) between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    return (2. * intersection) / (mask1.sum() + mask2.sum())

# Load original image
dicom_folder = r"C:\Users\sarra\Downloads\ORIG_3D_FSPGR_20_Average\ORIG_3D_FSPGR_20_Average"
reader = sitk.ImageSeriesReader()
reader.SetFileNames(reader.GetGDCMSeriesFileNames(dicom_folder))
raw_img_sitk = reader.Execute()

# Convert to float32 and rescale
raw_img_sitk = sitk.Cast(raw_img_sitk, sitk.sitkFloat32)
transformed = sitk.RescaleIntensity(raw_img_sitk, 0, 255)

# Apply Otsu thresholding on original image
original_mask = sitk.OtsuThreshold(transformed, 0, 1)
original_mask_array = sitk.GetArrayViewFromImage(original_mask)

# Perform N4 bias field correction
bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected = bias_corrector.Execute(transformed)

# Apply Otsu thresholding on bias-corrected image
corrected_mask = sitk.OtsuThreshold(corrected, 0, 1)
corrected_mask_array = sitk.GetArrayViewFromImage(corrected_mask)

# Convert images to NumPy arrays
original_array = sitk.GetArrayViewFromImage(raw_img_sitk)
corrected_array = sitk.GetArrayViewFromImage(corrected)

# Reference mask (can be a manually segmented ground truth, here we use the original mask)
reference_mask = original_mask_array  

# Compute Dice Coefficient
dsc_before = dice_coefficient(original_mask_array, reference_mask)
dsc_after = dice_coefficient(corrected_mask_array, reference_mask)

# Compute SSIM
ssim_before = ssim(original_array, original_mask_array, data_range=original_array.max() - original_array.min())
ssim_after = ssim(corrected_array, corrected_mask_array, data_range=corrected_array.max() - corrected_array.min())

# Print results
print(f"DSC Before Bias Correction: {dsc_before}")
print(f"DSC After Bias Correction: {dsc_after}")
print(f"SSIM Before Bias Correction: {ssim_before}")
print(f"SSIM After Bias Correction: {ssim_after}")

# Visualization
slice_idx = 21  # Select 22nd slice (index 21)

plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(original_array[slice_idx], cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(original_mask_array[slice_idx], cmap='gray')
plt.title("Original Threshold Mask")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(corrected_array[slice_idx], cmap='gray')
plt.title("Bias Corrected Image")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(corrected_mask_array[slice_idx], cmap='gray')
plt.title("Corrected Threshold Mask")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(corrected_array[slice_idx] - original_array[slice_idx], cmap='coolwarm')
plt.title("Difference (Corrected - Original)")
plt.axis("off")

plt.tight_layout()
plt.show()

