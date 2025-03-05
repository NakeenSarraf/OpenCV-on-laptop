import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

# Define the path to your DICOM folder
dicom_folder = r"C:\Users\sarra\Downloads\ORIG_3D_FSPGR_20_Average\ORIG_3D_FSPGR_20_Average"

# Create a DICOM series reader
reader = sitk.ImageSeriesReader()

# Get the list of DICOM file names in the specified folder
dicom_series = reader.GetGDCMSeriesFileNames(dicom_folder)

# Set the DICOM file names for the reader
reader.SetFileNames(dicom_series)

# Load the DICOM series into a SimpleITK image object
raw_img_sitk = reader.Execute()

# Check if the image was loaded successfully
if raw_img_sitk.GetSize() == (0, 0, 0):
    raise ValueError("Failed to load DICOM series. Check the folder path.")

# Convert to 32-bit float for processing
raw_img_sitk = sitk.Cast(raw_img_sitk, sitk.sitkFloat32)

# Perform N4 bias field correction
bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected_img = bias_corrector.Execute(raw_img_sitk)

# Rescale intensity to 0-255 for better contrast (optional)
corrected_rescaled = sitk.RescaleIntensity(corrected_img, 0, 255)

# Apply Otsu’s thresholding after bias correction
mask = sitk.OtsuThreshold(corrected_rescaled, 0, 1)

# Shrink the images **only in X and Y** (keep the Z dimension the same)
shrinkFactor = [1, 4, 4]  # Keep all slices (Z=1), shrink X & Y by 4
shrunken_corrected = sitk.Shrink(corrected_rescaled, shrinkFactor)
shrunken_mask = sitk.Shrink(mask, shrinkFactor)

# Convert to NumPy arrays
original_array = sitk.GetArrayViewFromImage(raw_img_sitk)
corrected_array = sitk.GetArrayViewFromImage(shrunken_corrected)
thresholded_array = sitk.GetArrayViewFromImage(shrunken_mask)

# Compute the difference: Original - Bias Corrected
difference_array = original_array - sitk.GetArrayViewFromImage(corrected_img)

# **Check the shape of the arrays**
print("Original shape:", original_array.shape)
print("Corrected shape:", corrected_array.shape)
print("Thresholded shape:", thresholded_array.shape)
print("Difference shape:", difference_array.shape)

# **Ensure slice 22 is accessible**
slice_idx = 21  # 22nd slice
if slice_idx >= corrected_array.shape[0]:
    raise IndexError(f"Slice index {slice_idx} is out of bounds for available slices {corrected_array.shape[0]}.")

# **Plot all images including the difference**
plt.figure(figsize=(16, 6))

plt.subplot(1, 4, 1)
plt.imshow(original_array[slice_idx], cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(corrected_array[slice_idx], cmap='gray')
plt.title("Bias Field Corrected (Shrunken)")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(thresholded_array[slice_idx], cmap='gray')
plt.title("Thresholded (Bias Corrected)")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(difference_array[slice_idx], cmap='coolwarm')
plt.title("Difference (Original - Corrected)")
plt.axis("off")

plt.show()
