"""
LESSON 1: DICOM Format Basics
Biomedical Image Processing - DICOM Module

Topics:
- What is DICOM?
- DICOM file structure
- Installing pydicom library
- Understanding DICOM metadata (tags)
"""

import numpy as np
import matplotlib.pyplot as plt

# pydicom is the main library for DICOM in Python
try:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import generate_uid
    print(f"pydicom version: {pydicom.__version__}")
    print("pydicom is ready!")
except ImportError:
    print("Please install pydicom: pip install pydicom")
    exit()

import datetime

# ============================================================
# 1. What is DICOM?
# ============================================================
"""
DICOM (Digital Imaging and Communications in Medicine) is the
international standard for medical images and related information.

Key Features:
- Universal Format: Used worldwide in hospitals and clinics
- Rich Metadata: Contains patient info, acquisition parameters, etc.
- Multi-modality: Supports CT, MRI, X-ray, Ultrasound, PET, etc.
- High Bit Depth: Typically 12-16 bits (vs 8-bit for JPEG/PNG)

Common File Extensions:
- .dcm - Standard DICOM extension
- .dicom - Alternative extension
- No extension - Many DICOM files have no extension
"""

# ============================================================
# 2. Creating a Synthetic DICOM Dataset
# ============================================================

# Create file meta information
file_meta = FileMetaDataset()
file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
file_meta.MediaStorageSOPInstanceUID = generate_uid()
file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

# Create the main dataset
ds = Dataset()
ds.file_meta = file_meta

# Patient Information
ds.PatientName = "Test^Patient"
ds.PatientID = "123456"
ds.PatientBirthDate = "19800101"
ds.PatientSex = "M"

# Study Information
ds.StudyDate = datetime.datetime.now().strftime("%Y%m%d")
ds.StudyDescription = "CT Chest"
ds.StudyInstanceUID = generate_uid()

# Series Information
ds.Modality = "CT"
ds.SeriesDescription = "Axial"
ds.SeriesInstanceUID = generate_uid()

# Image Information
ds.Rows = 512
ds.Columns = 512
ds.BitsAllocated = 16
ds.BitsStored = 12
ds.HighBit = 11
ds.PixelRepresentation = 1  # Signed
ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = "MONOCHROME2"

# Window/Level for CT
ds.WindowCenter = 40
ds.WindowWidth = 400

# Rescale values (Hounsfield Units for CT)
ds.RescaleIntercept = -1024
ds.RescaleSlope = 1

print("Synthetic DICOM dataset created!")
print(f"\nPatient: {ds.PatientName}")
print(f"Modality: {ds.Modality}")
print(f"Image Size: {ds.Rows} x {ds.Columns}")

# ============================================================
# 3. Exploring DICOM Tags
# ============================================================
print("\nDICOM Dataset Contents:")
print("=" * 50)

for elem in ds:
    if elem.tag.group != 0x7FE0:  # Skip pixel data
        print(f"{elem.tag} {elem.keyword}: {elem.value}")

# ============================================================
# 4. Bit Depth Comparison
# ============================================================
np.random.seed(42)

# 8-bit image (standard)
img_8bit = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

# 12-bit image (DICOM typical)
img_12bit = np.random.randint(0, 4096, (100, 100), dtype=np.uint16)

# 16-bit image (high precision DICOM)
img_16bit = np.random.randint(0, 65536, (100, 100), dtype=np.uint16)

print("\nBit Depth Comparison:")
print(f"8-bit:  min={img_8bit.min()}, max={img_8bit.max()}, levels={2**8}")
print(f"12-bit: min={img_12bit.min()}, max={img_12bit.max()}, levels={2**12}")
print(f"16-bit: min={img_16bit.min()}, max={img_16bit.max()}, levels={2**16}")

# Visualize the effect of bit depth on gradient
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Create gradients with different bit depths
gradient_8bit = np.tile(np.linspace(0, 255, 256).astype(np.uint8), (50, 1))
gradient_12bit = np.tile(np.linspace(0, 4095, 256).astype(np.uint16), (50, 1))
gradient_16bit = np.tile(np.linspace(0, 65535, 256).astype(np.uint16), (50, 1))

axes[0].imshow(gradient_8bit, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('8-bit (256 levels)\nStandard Images')
axes[0].axis('off')

axes[1].imshow(gradient_12bit, cmap='gray', vmin=0, vmax=4095)
axes[1].set_title('12-bit (4,096 levels)\nTypical DICOM')
axes[1].axis('off')

axes[2].imshow(gradient_16bit, cmap='gray', vmin=0, vmax=65535)
axes[2].set_title('16-bit (65,536 levels)\nHigh Precision DICOM')
axes[2].axis('off')

plt.suptitle('Bit Depth Comparison - More bits = More precision', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 5. Hounsfield Units (CT Images)
# ============================================================
"""
CT images use Hounsfield Units (HU) to represent tissue density:

| Tissue      | HU Value      |
|-------------|---------------|
| Air         | -1000         |
| Lung        | -500          |
| Fat         | -100 to -50   |
| Water       | 0             |
| Soft Tissue | +40 to +80    |
| Bone        | +400 to +1000 |

Conversion Formula:
HU = pixel_value * RescaleSlope + RescaleIntercept
"""

# Simulate a CT image with different tissues
ct_image = np.zeros((200, 200), dtype=np.int16)

# Background (air): -1000 HU
ct_image[:, :] = -1000

# Body outline (soft tissue): +40 HU
y, x = np.ogrid[:200, :200]
body_mask = ((x - 100)**2 + (y - 100)**2) < 80**2
ct_image[body_mask] = 40

# Lung regions: -500 HU
lung_left = ((x - 70)**2 + (y - 100)**2) < 25**2
lung_right = ((x - 130)**2 + (y - 100)**2) < 25**2
ct_image[lung_left | lung_right] = -500

# Spine (bone): +700 HU
spine_mask = ((x - 100)**2 + (y - 150)**2) < 15**2
ct_image[spine_mask] = 700

print("\nSimulated CT Image (Hounsfield Units):")
print(f"  Min HU: {ct_image.min()}")
print(f"  Max HU: {ct_image.max()}")

# Display the simulated CT with proper windowing
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Full range
im1 = axes[0].imshow(ct_image, cmap='gray', vmin=-1000, vmax=1000)
axes[0].set_title('Full Range\n(-1000 to +1000 HU)')
plt.colorbar(im1, ax=axes[0], label='HU')

# Soft tissue window
im2 = axes[1].imshow(ct_image, cmap='gray', vmin=-160, vmax=240)
axes[1].set_title('Soft Tissue Window\n(W:400, L:40)')
plt.colorbar(im2, ax=axes[1], label='HU')

# Lung window
im3 = axes[2].imshow(ct_image, cmap='gray', vmin=-1200, vmax=200)
axes[2].set_title('Lung Window\n(W:1400, L:-500)')
plt.colorbar(im3, ax=axes[2], label='HU')

for ax in axes:
    ax.axis('off')

plt.suptitle('Same CT Image - Different Windows', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
What we learned:
1. DICOM is the standard format for medical images
2. DICOM files contain metadata (tags) and pixel data
3. pydicom is the Python library for working with DICOM
4. Medical images use higher bit depth (12-16 bit) than standard images
5. CT images use Hounsfield Units for tissue density
6. Window/Level settings control how we view the image
""")
