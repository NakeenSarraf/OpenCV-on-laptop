import SimpleITK as sitk
import os

# Define input and output file paths
nifti_path = r"C:\Users\sarra\Downloads\Opencv\biasfield first.nii.gz" 
output_nifti_path = r"C:\Users\sarra\Downloads\bfc_int_first.nii.gz"

# Load the NIfTI image
image = sitk.ReadImage(nifti_path)

# Convert to float for processing
image = sitk.Cast(image, sitk.sitkFloat32)

# Get min and max intensity values
stats = sitk.Statistics C:\Users\sarra\Downloads\OpencvImageFilter()
stats.Execute(image)
min_intensity = stats.GetMinimum()
max_intensity = stats.GetMaximum()

print(f"Original Intensity Range: Min = {min_intensity}, Max = {max_intensity}")

# Adjust intensity by rescaling to a new range (e.g., 0 to 255)
adjusted_image = sitk.RescaleIntensity(image, 0, 255)

# Get min and max after rescaling
stats.Execute(adjusted_image)
min_rescaled = stats.GetMinimum()
max_rescaled = stats.GetMaximum()

print(f"Rescaled Intensity Range: Min = {min_rescaled}, Max = {max_rescaled}")

# Save the modified image
sitk.WriteImage(adjusted_image, output_nifti_path)

print(f"Adjusted intensity NIfTI saved at: {output_nifti_path}")
