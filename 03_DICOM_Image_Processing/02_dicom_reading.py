"""
LESSON 2: Reading and Displaying DICOM Images
Biomedical Image Processing - DICOM Module

Topics:
- Reading DICOM files with pydicom
- Extracting pixel data
- Applying rescale slope and intercept
- Displaying DICOM images properly
"""

import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid
import os

print("Libraries imported successfully!")

# ============================================================
# 1. Creating a Sample DICOM File
# ============================================================

def create_sample_dicom(filename, modality='CT'):
    """Create a sample DICOM file for demonstration."""

    # Create file meta
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()  # Required for valid DICOM

    # Create dataset
    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Add preamble (128 bytes of zeros + 'DICM')
    ds.preamble = b'\x00' * 128

    # Patient info
    ds.PatientName = "Test^Patient"
    ds.PatientID = "12345"
    ds.PatientBirthDate = "19850315"
    ds.PatientSex = "F"

    # Study info
    ds.StudyDate = "20240115"
    ds.StudyTime = "103000"
    ds.StudyDescription = "CT Abdomen"
    ds.StudyInstanceUID = generate_uid()

    # Series info
    ds.Modality = modality
    ds.SeriesDescription = "Axial"
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 1

    # Image info
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceNumber = 1
    ds.ImagePositionPatient = [0, 0, 0]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [0.5, 0.5]  # 0.5mm pixels
    ds.SliceThickness = 2.5  # 2.5mm slice

    # Pixel data properties
    ds.Rows = 256
    ds.Columns = 256
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 1  # Signed
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # Rescale for Hounsfield Units
    ds.RescaleIntercept = -1024
    ds.RescaleSlope = 1

    # Window/Level
    ds.WindowCenter = 40
    ds.WindowWidth = 400

    # Create synthetic CT-like image
    np.random.seed(42)

    # Start with noise (stored values, not HU)
    pixel_array = np.random.randint(900, 1100, (256, 256), dtype=np.int16)

    # Add body outline (soft tissue ~40 HU -> stored ~1064)
    y, x = np.ogrid[:256, :256]
    body = ((x - 128)**2 + (y - 128)**2) < 100**2
    pixel_array[body] = np.random.randint(1040, 1088, np.sum(body))

    # Add lung regions (air ~-500 HU -> stored ~524)
    lung_l = ((x - 90)**2 + (y - 128)**2) < 35**2
    lung_r = ((x - 166)**2 + (y - 128)**2) < 35**2
    pixel_array[lung_l | lung_r] = np.random.randint(400, 600, np.sum(lung_l | lung_r))

    # Add spine (bone ~500 HU -> stored ~1524)
    spine = ((x - 128)**2 + (y - 180)**2) < 20**2
    pixel_array[spine] = np.random.randint(1400, 1700, np.sum(spine))

    # Add liver region
    liver = ((x - 160)**2 / 900 + (y - 140)**2 / 400) < 1
    pixel_array[liver] = np.random.randint(1060, 1100, np.sum(liver))

    ds.PixelData = pixel_array.tobytes()

    # Save with proper DICOM format
    ds.save_as(filename, write_like_original=False)
    print(f"Created: {filename}")

    return ds

# Create sample DICOM
sample_ds = create_sample_dicom('sample_ct.dcm')

# ============================================================
# 2. Reading DICOM Files
# ============================================================

# Read the DICOM file
ds = pydicom.dcmread('sample_ct.dcm')

print("\nDICOM File Loaded Successfully!")
print("=" * 50)

# Display patient information
print("\nPATIENT INFORMATION")
print("-" * 30)
print(f"Patient Name: {ds.PatientName}")
print(f"Patient ID: {ds.PatientID}")
print(f"Birth Date: {ds.PatientBirthDate}")
print(f"Sex: {ds.PatientSex}")

# Display study and series information
print("\nSTUDY INFORMATION")
print("-" * 30)
print(f"Study Date: {ds.StudyDate}")
print(f"Study Description: {ds.StudyDescription}")
print(f"Modality: {ds.Modality}")
print(f"Series Description: {ds.SeriesDescription}")

# Display image properties
print("\nIMAGE PROPERTIES")
print("-" * 30)
print(f"Rows (Height): {ds.Rows}")
print(f"Columns (Width): {ds.Columns}")
print(f"Bits Allocated: {ds.BitsAllocated}")
print(f"Bits Stored: {ds.BitsStored}")
print(f"Pixel Spacing: {ds.PixelSpacing} mm")
print(f"Slice Thickness: {ds.SliceThickness} mm")

# ============================================================
# 3. Extracting Pixel Data
# ============================================================

# Get pixel data
pixel_array = ds.pixel_array

print("\nPIXEL DATA")
print("-" * 30)
print(f"Shape: {pixel_array.shape}")
print(f"Data Type: {pixel_array.dtype}")
print(f"Min Value (stored): {pixel_array.min()}")
print(f"Max Value (stored): {pixel_array.max()}")

# ============================================================
# 4. Applying Rescale Slope and Intercept
# ============================================================

# Get rescale parameters
slope = ds.RescaleSlope
intercept = ds.RescaleIntercept

print(f"\nRescale Slope: {slope}")
print(f"Rescale Intercept: {intercept}")

# Apply rescaling to get Hounsfield Units
hu_image = pixel_array * slope + intercept

print(f"\nAfter Rescaling (Hounsfield Units):")
print(f"Min HU: {hu_image.min()}")
print(f"Max HU: {hu_image.max()}")

# Helper function to apply rescaling
def apply_rescale(ds):
    """Apply rescale slope and intercept to pixel data."""
    pixel_array = ds.pixel_array.astype(float)

    # Get rescale parameters (default to 1 and 0 if not present)
    slope = getattr(ds, 'RescaleSlope', 1)
    intercept = getattr(ds, 'RescaleIntercept', 0)

    return pixel_array * slope + intercept

# Apply rescaling
hu_image = apply_rescale(ds)
print(f"\nRescaled image shape: {hu_image.shape}")
print(f"HU range: [{hu_image.min():.0f}, {hu_image.max():.0f}]")

# ============================================================
# 5. Displaying DICOM Images
# ============================================================

# Simple display - but this won't look good!
plt.figure(figsize=(8, 8))
plt.imshow(hu_image, cmap='gray')
plt.title('DICOM Image - Auto Scaling\n(Notice poor contrast)')
plt.colorbar(label='Hounsfield Units')
plt.axis('off')
plt.show()

# Better display with Window/Level
window_center = float(ds.WindowCenter)
window_width = float(ds.WindowWidth)

# Calculate display range
vmin = window_center - window_width / 2
vmax = window_center + window_width / 2

print(f"\nWindow Center: {window_center}")
print(f"Window Width: {window_width}")
print(f"Display Range: [{vmin}, {vmax}] HU")

plt.figure(figsize=(8, 8))
plt.imshow(hu_image, cmap='gray', vmin=vmin, vmax=vmax)
plt.title(f'DICOM Image with Window/Level\nW:{window_width} L:{window_center}')
plt.colorbar(label='Hounsfield Units')
plt.axis('off')
plt.show()

# ============================================================
# 6. Complete DICOM Reading Function
# ============================================================

def read_dicom(filepath, apply_window=True):
    """
    Read a DICOM file and return processed image with metadata.

    Parameters:
    -----------
    filepath : str
        Path to DICOM file
    apply_window : bool
        Whether to apply window/level from DICOM tags

    Returns:
    --------
    dict with 'image', 'metadata', 'window_center', 'window_width'
    """
    # Read file
    ds = pydicom.dcmread(filepath)

    # Get pixel array
    pixel_array = ds.pixel_array.astype(float)

    # Apply rescaling
    slope = getattr(ds, 'RescaleSlope', 1)
    intercept = getattr(ds, 'RescaleIntercept', 0)
    image = pixel_array * slope + intercept

    # Get window/level
    wc = getattr(ds, 'WindowCenter', None)
    ww = getattr(ds, 'WindowWidth', None)

    # Handle multiple windows (some DICOM have lists)
    if isinstance(wc, pydicom.multival.MultiValue):
        wc = float(wc[0])
    elif wc is not None:
        wc = float(wc)

    if isinstance(ww, pydicom.multival.MultiValue):
        ww = float(ww[0])
    elif ww is not None:
        ww = float(ww)

    # Collect metadata
    metadata = {
        'patient_name': str(getattr(ds, 'PatientName', 'Unknown')),
        'patient_id': str(getattr(ds, 'PatientID', 'Unknown')),
        'modality': str(getattr(ds, 'Modality', 'Unknown')),
        'study_date': str(getattr(ds, 'StudyDate', 'Unknown')),
        'study_description': str(getattr(ds, 'StudyDescription', '')),
        'rows': ds.Rows,
        'columns': ds.Columns,
        'pixel_spacing': getattr(ds, 'PixelSpacing', None),
        'slice_thickness': getattr(ds, 'SliceThickness', None),
    }

    return {
        'image': image,
        'metadata': metadata,
        'window_center': wc,
        'window_width': ww,
        'dataset': ds  # Keep original for advanced use
    }

# Test the function
result = read_dicom('sample_ct.dcm')

print("\nDICOM loaded successfully!")
print(f"Image shape: {result['image'].shape}")
print(f"Modality: {result['metadata']['modality']}")
print(f"Window: W={result['window_width']}, L={result['window_center']}")

# ============================================================
# 7. Checking for Required Tags
# ============================================================

def check_dicom_tags(ds):
    """Check for important DICOM tags and report status."""

    important_tags = [
        ('PatientName', 'Patient Name'),
        ('PatientID', 'Patient ID'),
        ('Modality', 'Modality'),
        ('StudyDate', 'Study Date'),
        ('Rows', 'Image Rows'),
        ('Columns', 'Image Columns'),
        ('BitsAllocated', 'Bits Allocated'),
        ('PixelSpacing', 'Pixel Spacing'),
        ('SliceThickness', 'Slice Thickness'),
        ('RescaleSlope', 'Rescale Slope'),
        ('RescaleIntercept', 'Rescale Intercept'),
        ('WindowCenter', 'Window Center'),
        ('WindowWidth', 'Window Width'),
    ]

    print("\nDICOM Tag Check")
    print("=" * 50)

    for tag, name in important_tags:
        if hasattr(ds, tag):
            value = getattr(ds, tag)
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 30:
                str_value = str_value[:27] + "..."
            print(f"[OK] {name:20s}: {str_value}")
        else:
            print(f"[--] {name:20s}: Not found")

check_dicom_tags(ds)

# ============================================================
# Clean up
# ============================================================
if os.path.exists('sample_ct.dcm'):
    os.remove('sample_ct.dcm')
    print("\nSample file cleaned up.")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
What we learned:
1. Use pydicom.dcmread() to read DICOM files
2. Access pixel data with ds.pixel_array
3. Apply RescaleSlope and RescaleIntercept to get actual values
4. Use Window/Level for proper display
5. Always check if tags exist before accessing them
6. Extract metadata for patient and study information
""")
