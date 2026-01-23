"""
LESSON 4: Working with DICOM Series (3D Volumes)
Biomedical Image Processing - DICOM Module

Topics:
- Understanding DICOM series structure
- Loading multiple slices as a 3D volume
- Sorting slices correctly
- Visualizing 3D CT/MRI data
"""

import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid
import os
import shutil

print("Libraries imported successfully!")

# ============================================================
# 1. Create a 3D Phantom
# ============================================================

def create_3d_phantom(size=64, num_slices=20):
    """
    Create a 3D phantom with anatomical-like structures.
    Returns pixel values in stored format (add 1024 for HU).
    """
    volume = np.zeros((num_slices, size, size), dtype=np.int16)

    # Background: Air (-1000 HU -> stored 24)
    volume[:] = 24

    z, y, x = np.ogrid[:num_slices, :size, :size]
    center = size // 2
    z_center = num_slices // 2

    # Body ellipsoid (soft tissue ~40 HU -> stored 1064)
    body = ((x - center)**2 / (center*0.8)**2 +
            (y - center)**2 / (center*0.9)**2 +
            (z - z_center)**2 / (z_center*0.9)**2) < 1
    volume[body] = 1064 + np.random.randint(-20, 20, np.sum(body))

    # Spine (bone ~500 HU -> stored 1524)
    spine_radius = size // 10
    spine = ((x - center)**2 + (y - int(center*1.4))**2) < spine_radius**2
    spine = spine & body
    volume[spine] = 1524 + np.random.randint(-50, 50, np.sum(spine))

    # Lungs (air ~-500 HU -> stored 524)
    lung_slices = (z > num_slices*0.2) & (z < num_slices*0.8)
    lung_left = ((x - int(center*0.6))**2 + (y - center)**2) < (size//6)**2
    lung_right = ((x - int(center*1.4))**2 + (y - center)**2) < (size//6)**2
    lungs = lung_slices & (lung_left | lung_right) & body
    volume[lungs] = 524 + np.random.randint(-50, 50, np.sum(lungs))

    # Heart (blood ~50 HU -> stored 1074)
    heart_slices = (z > num_slices*0.3) & (z < num_slices*0.7)
    heart = ((x - center)**2 + (y - int(center*0.9))**2) < (size//8)**2
    heart = heart_slices & heart & body
    volume[heart] = 1074 + np.random.randint(-10, 10, np.sum(heart))

    return volume

# Create 3D phantom
np.random.seed(42)
phantom_3d = create_3d_phantom(size=64, num_slices=20)
print(f"3D Phantom shape: {phantom_3d.shape}")
print(f"Value range: [{phantom_3d.min()}, {phantom_3d.max()}]")

# ============================================================
# 2. Create DICOM Series
# ============================================================

def create_dicom_series(volume, output_dir, series_description="CT Series"):
    """Create a DICOM series from a 3D volume."""
    os.makedirs(output_dir, exist_ok=True)

    num_slices, rows, cols = volume.shape

    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_of_ref_uid = generate_uid()

    slice_thickness = 2.5
    pixel_spacing = [1.0, 1.0]

    for i in range(num_slices):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        ds = Dataset()
        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.preamble = b'\x00' * 128

        # Patient
        ds.PatientName = "Phantom^3D"
        ds.PatientID = "PHANTOM001"
        ds.PatientBirthDate = "19900101"
        ds.PatientSex = "O"

        # Study
        ds.StudyDate = "20240120"
        ds.StudyTime = "120000"
        ds.StudyDescription = "3D Phantom Study"
        ds.StudyInstanceUID = study_uid
        ds.StudyID = "1"

        # Series
        ds.Modality = "CT"
        ds.SeriesDescription = series_description
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = 1

        # Frame of Reference
        ds.FrameOfReferenceUID = frame_of_ref_uid
        ds.PositionReferenceIndicator = ""

        # Image
        ds.SOPClassUID = pydicom.uid.CTImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = i + 1
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]

        # Position and Orientation
        z_position = i * slice_thickness
        ds.ImagePositionPatient = [0, 0, z_position]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.SliceLocation = z_position
        ds.SliceThickness = slice_thickness
        ds.PixelSpacing = pixel_spacing

        # Pixel Data
        ds.Rows = rows
        ds.Columns = cols
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        # Rescale
        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1

        # Window
        ds.WindowCenter = 40
        ds.WindowWidth = 400

        ds.PixelData = volume[i].tobytes()

        filename = os.path.join(output_dir, f"slice_{i+1:03d}.dcm")
        pydicom.dcmwrite(filename, ds, write_like_original=False)

    print(f"Created {num_slices} DICOM files in '{output_dir}'")
    return output_dir

# Create DICOM series
series_dir = create_dicom_series(phantom_3d, "sample_series", "Axial CT")

# ============================================================
# 3. Load DICOM Series
# ============================================================

def load_dicom_series(directory):
    """Load all DICOM files from a directory and return as 3D volume."""
    dicom_files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            ds = pydicom.dcmread(filepath)
            dicom_files.append(ds)
        except:
            continue

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {directory}")

    print(f"Found {len(dicom_files)} DICOM files")

    # Sort by SliceLocation or InstanceNumber
    try:
        dicom_files.sort(key=lambda x: float(x.SliceLocation))
        print("Sorted by SliceLocation")
    except:
        dicom_files.sort(key=lambda x: int(x.InstanceNumber))
        print("Sorted by InstanceNumber")

    first_slice = dicom_files[0]
    rows = first_slice.Rows
    cols = first_slice.Columns
    num_slices = len(dicom_files)

    print(f"Volume dimensions: {num_slices} x {rows} x {cols}")

    volume = np.zeros((num_slices, rows, cols), dtype=np.float32)

    for i, ds in enumerate(dicom_files):
        pixel_array = ds.pixel_array.astype(float)
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        volume[i] = pixel_array * slope + intercept

    return volume, dicom_files

# Load the series
volume, slices = load_dicom_series("sample_series")
print(f"\nLoaded volume shape: {volume.shape}")
print(f"HU range: [{volume.min():.0f}, {volume.max():.0f}]")

# ============================================================
# 4. Orthogonal Views
# ============================================================

def show_orthogonal_views(volume, slice_indices=None, window_center=40, window_width=400):
    """Show axial, sagittal, and coronal views of a 3D volume."""
    nz, ny, nx = volume.shape

    if slice_indices is None:
        slice_indices = (nz // 2, ny // 2, nx // 2)

    ax_idx, sag_idx, cor_idx = slice_indices

    vmin = window_center - window_width / 2
    vmax = window_center + window_width / 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial view
    axes[0].imshow(volume[ax_idx, :, :], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Axial (Slice {ax_idx + 1}/{nz})')
    axes[0].axis('off')

    # Sagittal view
    axes[1].imshow(volume[:, :, sag_idx], cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title(f'Sagittal (X = {sag_idx})')
    axes[1].axis('off')

    # Coronal view
    axes[2].imshow(volume[:, cor_idx, :], cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
    axes[2].set_title(f'Coronal (Y = {cor_idx})')
    axes[2].axis('off')

    plt.suptitle('Orthogonal Views', fontsize=14)
    plt.tight_layout()
    plt.show()

show_orthogonal_views(volume)

# ============================================================
# 5. Slice Montage
# ============================================================

def show_slice_montage(volume, num_cols=5, window_center=40, window_width=400):
    """Show all slices as a montage."""
    num_slices = volume.shape[0]
    num_rows = (num_slices + num_cols - 1) // num_cols

    vmin = window_center - window_width / 2
    vmax = window_center + window_width / 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, 3*num_rows))
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < num_slices:
            axes[i].imshow(volume[i], cmap='gray', vmin=vmin, vmax=vmax)
            axes[i].set_title(f'Slice {i+1}', fontsize=10)
        axes[i].axis('off')

    plt.suptitle('All Slices Montage', fontsize=14)
    plt.tight_layout()
    plt.show()

show_slice_montage(volume, num_cols=5)

# ============================================================
# 6. Maximum Intensity Projection (MIP)
# ============================================================

def maximum_intensity_projection(volume):
    """Create Maximum Intensity Projections along all three axes."""
    mip_axial = np.max(volume, axis=0)
    mip_sagittal = np.max(volume, axis=2)
    mip_coronal = np.max(volume, axis=1)
    return mip_axial, mip_sagittal, mip_coronal

mip_ax, mip_sag, mip_cor = maximum_intensity_projection(volume)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

vmin, vmax = -200, 800

axes[0].imshow(mip_ax, cmap='gray', vmin=vmin, vmax=vmax)
axes[0].set_title('MIP - Axial')
axes[0].axis('off')

axes[1].imshow(mip_sag, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
axes[1].set_title('MIP - Sagittal')
axes[1].axis('off')

axes[2].imshow(mip_cor, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
axes[2].set_title('MIP - Coronal')
axes[2].axis('off')

plt.suptitle('Maximum Intensity Projections', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# Clean up
# ============================================================
if os.path.exists('sample_series'):
    shutil.rmtree('sample_series')
    print("\nSample series cleaned up.")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
What we learned:
1. DICOM series consist of multiple 2D slices forming a 3D volume
2. Sort slices by SliceLocation or InstanceNumber
3. Extract voxel spacing from PixelSpacing and SliceThickness
4. View orthogonal planes: Axial, Sagittal, Coronal
5. MIP (Maximum Intensity Projection) for vessel/bone visualization
""")
