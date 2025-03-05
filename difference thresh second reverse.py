import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np

# Define paths to your DICOM folder and NIfTI file
dicom_folder = r"C:\Users\sarra\Downloads\ORIG_3D_FSPGR_20_Average\ORIG_3D_FSPGR_20_Average"
corrected_nifti = r"C:\Users\sarra\Downloads\Opencv\niftyfiles\biasfield first.nii"  # Ensure file extension is correct

# Check if the corrected NIfTI file exists
if not os.path.exists(corrected_nifti):
    raise FileNotFoundError(f"❌ File not found: {corrected_nifti}")

# Create a DICOM series reader
reader = sitk.ImageSeriesReader()

# Get the list of DICOM file names in the specified folder
dicom_series = reader.GetGDCMSeriesFileNames(dicom_folder)

# Set the DICOM file names for the reader
reader.SetFileNames(dicom_series)

# Load the DICOM series into a SimpleITK image object
original_img = reader.Execute()

# Check if the image was loaded successfully
if original_img.GetSize() == (0, 0, 0):
    raise ValueError("❌ Failed to load DICOM series. Check the folder path.")

# Convert to 32-bit float for processing
original_img = sitk.Cast(original_img, sitk.sitkFloat32)

# Read the corrected NIfTI file
corrected_img = sitk.ReadImage(corrected_nifti)

# Convert the corrected NIfTI image to float
corrected_img = sitk.Cast(corrected_img, sitk.sitkFloat32)

# Rescale intensity of the corrected image (optional, for better visualization)
corrected_rescaled = sitk.RescaleIntensity(corrected_img, 0, 255)

# Convert to NumPy arrays for easier plotting
original_array = sitk.GetArrayViewFromImage(original_img)
corrected_array = sitk.GetArrayViewFromImage(corrected_rescaled)

# Compute the difference (Original - Corrected)
difference_array = corrected_array - original_array

# **Check the shape of the arrays**
print("Original shape:", original_array.shape)
print("Corrected shape:", corrected_array.shape)
print("Difference shape:", difference_array.shape)

# Adjust slice index to be within the available range
slice_idx = 21  # 22nd slice (index 21)

# Ensure the slice index is within the bounds of the image
if slice_idx >= original_array.shape[0]:
    slice_idx = original_array.shape[0] - 1  # Use the last available slice

# Plot all images including the difference
plt.figure(figsize=(16, 6))

# Original image
plt.subplot(1, 4, 1)
plt.imshow(original_array[slice_idx], cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Corrected image
plt.subplot(1, 4, 2)
plt.imshow(corrected_array[slice_idx], cmap='gray')
plt.title("Bias Field Corrected")
plt.axis("off")

# Difference between original and corrected
plt.subplot(1, 4, 3)
plt.imshow(difference_array[slice_idx], cmap='coolwarm')
plt.title("Difference (Original - Corrected)")
plt.axis("off")

plt.show()
