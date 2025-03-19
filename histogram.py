import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# Load the placenta image 
dicom_folder = r"C:\Users\sarra\Downloads\ORIG_3D_FSPGR_20_Average\ORIG_3D_FSPGR_20_Average"
reader = sitk.ImageSeriesReader()
reader.SetFileNames(reader.GetGDCMSeriesFileNames(dicom_folder))
image_sitk = reader.Execute()

# Convert to NumPy
image_array = sitk.GetArrayFromImage(image_sitk)

# Select the 22th slice (index 21)
slice_idx = 21
image_slice = image_array[slice_idx]

# Compute histogram
hist, bins = np.histogram(image_slice.flatten(), bins=256, range=[image_slice.min(), image_slice.max()])

# Apply Otsu's threshold
otsu_filter = sitk.OtsuThresholdImageFilter()
otsu_threshold = otsu_filter.Execute(sitk.GetImageFromArray(image_slice))
threshold_value = otsu_filter.GetThreshold()

# Create a figure with two subplots: one for the image and one for the histogram
plt.figure(figsize=(12, 6))

# Display slide 44
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
plt.imshow(image_slice, cmap="gray")  # Display the image in grayscale
plt.title(f"DICOM Image (Slide {slice_idx + 1})")
plt.axis("off")  # Hide the axis

# Plot histogram with Otsu's threshold
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.plot(bins[:-1], hist, label="Histogram", color="blue")
plt.axvline(threshold_value, color="red", linestyle="dashed", label=f"Otsu Threshold: {threshold_value:.2f}")
plt.title("Histogram of Pixel Intensities with Otsu's Threshold")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
