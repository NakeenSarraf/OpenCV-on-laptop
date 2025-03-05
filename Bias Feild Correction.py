import SimpleITK as sitk
import os

# Define the path to your DICOM folder
dicom_folder = r"C:\Users\sarra\Downloads\ORIG_3D_FSPGR_20_Average\ORIG_3D_FSPGR_20_Average"

# Define output file paths for NIfTI files
bias_corrected_nifti = os.path.join(dicom_folder, "bias_corrected.nii.gz")
thresholded_nifti = os.path.join(dicom_folder, "thresholded_mask.nii.gz")

# Create a DICOM series reader
reader = sitk.ImageSeriesReader()

# Get the list of DICOM file names in the specified folder
dicom_series = reader.GetGDCMSeriesFileNames(dicom_folder)

# Set the DICOM file names for the reader
reader.SetFileNames(dicom_series)

# Load the DICOM series into a SimpleITK image object
raw_img_sitk = reader.Execute()

# Convert the image to a 32-bit float for processing
raw_img_sitk = sitk.Cast(raw_img_sitk, sitk.sitkFloat32)

# Perform N4 bias field correction
bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected_img = bias_corrector.Execute(raw_img_sitk)

# Apply Otsu thresholding on the bias-corrected image
threshold_mask = sitk.OtsuThreshold(corrected_img, 0, 1)

# Save the corrected image and segmentation mask as NIfTI files
sitk.WriteImage(corrected_img, bias_corrected_nifti)
sitk.WriteImage(threshold_mask, thresholded_nifti)

print(f"Bias-corrected image saved at: {bias_corrected_nifti}")
print(f"Thresholded segmentation mask saved at: {thresholded_nifti}")