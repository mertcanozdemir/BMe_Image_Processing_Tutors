"""
LESSON 5: DICOM Metadata Deep Dive
Biomedical Image Processing - DICOM Module

Topics:
- Understanding DICOM tag structure
- Reading and modifying metadata
- Working with sequences
- Extracting acquisition parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid
from pydicom.tag import Tag
from pydicom.datadict import keyword_for_tag, tag_for_keyword
from pydicom.dataelem import DataElement
from pydicom.sequence import Sequence
from datetime import datetime

print("Libraries imported successfully!")

# ============================================================
# 1. Create Sample DICOM with Rich Metadata
# ============================================================

def create_sample_dicom():
    """Create a sample DICOM with comprehensive metadata."""
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

    # Patient Information
    ds.PatientName = "Doe^John^Mr"
    ds.PatientID = "PAT12345"
    ds.PatientBirthDate = "19800515"
    ds.PatientSex = "M"
    ds.PatientAge = "044Y"
    ds.PatientWeight = 75.5

    # Study Information
    ds.StudyDate = "20240120"
    ds.StudyTime = "143052"
    ds.StudyDescription = "CT Chest with Contrast"
    ds.StudyInstanceUID = generate_uid()
    ds.AccessionNumber = "ACC001"
    ds.ReferringPhysicianName = "Smith^Jane^Dr"
    ds.InstitutionName = "City Hospital"
    ds.StationName = "CT_SCANNER_01"
    ds.Manufacturer = "Medical Imaging Co."
    ds.ManufacturerModelName = "UltraCT 5000"

    # Series Information
    ds.Modality = "CT"
    ds.SeriesDescription = "Axial 2.5mm"
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 3
    ds.BodyPartExamined = "CHEST"

    # Acquisition Parameters
    ds.KVP = 120
    ds.XRayTubeCurrent = 250
    ds.ExposureTime = 500
    ds.Exposure = 125
    ds.ConvolutionKernel = "STANDARD"
    ds.SliceThickness = 2.5
    ds.ContrastBolusAgent = "Omnipaque 350"

    # Image Information
    ds.Rows = 512
    ds.Columns = 512
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = [0.684, 0.684]
    ds.RescaleIntercept = -1024
    ds.RescaleSlope = 1
    ds.WindowCenter = [40, -500]
    ds.WindowWidth = [400, 1500]

    # Position
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceNumber = 42
    ds.ImagePositionPatient = [-175.0, -175.0, 105.0]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.SliceLocation = 105.0
    ds.FrameOfReferenceUID = generate_uid()

    # Create pixel data
    np.random.seed(42)
    pixel_array = np.random.randint(900, 1100, (512, 512), dtype=np.int16)
    ds.PixelData = pixel_array.tobytes()

    return ds

ds = create_sample_dicom()
print("Sample DICOM created!")

# ============================================================
# 2. Exploring Tags
# ============================================================

print("\n" + "=" * 70)
print("ALL DICOM TAGS")
print("=" * 70)
print(f"{'Tag':<15} {'VR':<5} {'Keyword':<35} {'Value'}")
print("-" * 70)

for elem in ds:
    if elem.tag.group != 0x7FE0:  # Skip pixel data
        tag_str = f"({elem.tag.group:04X},{elem.tag.element:04X})"
        vr = elem.VR
        keyword = elem.keyword
        value = str(elem.value)
        if len(value) > 30:
            value = value[:27] + "..."
        print(f"{tag_str:<15} {vr:<5} {keyword:<35} {value}")

# ============================================================
# 3. Different Ways to Access Tags
# ============================================================

print("\n" + "=" * 50)
print("ACCESSING TAGS - Different Methods")
print("=" * 50)

# Method 1: By keyword
print(f"\n1. By keyword: ds.PatientName = {ds.PatientName}")

# Method 2: By tag tuple
print(f"2. By tag tuple: ds[0x0010, 0x0010] = {ds[0x0010, 0x0010].value}")

# Method 3: By Tag object
tag = Tag(0x0010, 0x0010)
print(f"3. By Tag object: ds[Tag(0x0010, 0x0010)] = {ds[tag].value}")

# Method 4: Finding tag from keyword
patient_name_tag = tag_for_keyword('PatientName')
print(f"4. tag_for_keyword('PatientName') = {patient_name_tag}")

# Method 5: Finding keyword from tag
keyword = keyword_for_tag(0x00100010)
print(f"5. keyword_for_tag(0x00100010) = {keyword}")

# ============================================================
# 4. Extract Specific Information
# ============================================================

def extract_patient_info(ds):
    """Extract patient-related information."""
    info = {}
    tags = [
        ('PatientName', 'Name'),
        ('PatientID', 'ID'),
        ('PatientBirthDate', 'Birth Date'),
        ('PatientSex', 'Sex'),
        ('PatientAge', 'Age'),
        ('PatientWeight', 'Weight (kg)'),
    ]
    for tag, label in tags:
        value = getattr(ds, tag, None)
        if value is not None:
            info[label] = str(value)
    return info

def extract_acquisition_params(ds):
    """Extract CT acquisition parameters."""
    params = {}
    tags = [
        ('KVP', 'Tube Voltage (kVp)'),
        ('XRayTubeCurrent', 'Tube Current (mA)'),
        ('ExposureTime', 'Exposure Time (ms)'),
        ('Exposure', 'Exposure (mAs)'),
        ('SliceThickness', 'Slice Thickness (mm)'),
        ('ConvolutionKernel', 'Reconstruction Kernel'),
        ('ContrastBolusAgent', 'Contrast Agent'),
    ]
    for tag, label in tags:
        value = getattr(ds, tag, None)
        if value is not None:
            params[label] = value
    return params

patient_info = extract_patient_info(ds)
print("\nPATIENT INFORMATION")
print("=" * 40)
for key, value in patient_info.items():
    print(f"  {key}: {value}")

acq_params = extract_acquisition_params(ds)
print("\nACQUISITION PARAMETERS")
print("=" * 40)
for key, value in acq_params.items():
    print(f"  {key}: {value}")

# ============================================================
# 5. Modifying Metadata
# ============================================================

print("\n" + "=" * 50)
print("MODIFYING METADATA")
print("=" * 50)

ds_copy = create_sample_dicom()

print(f"Before: PatientName = {ds_copy.PatientName}")
ds_copy.PatientName = "Anonymous^Patient"
print(f"After: PatientName = {ds_copy.PatientName}")

# Adding new tag using DataElement
new_elem = DataElement(
    tag=Tag(0x0008, 0x0080),
    VR='LO',
    value='New Institution Name'
)
ds_copy.add(new_elem)
print(f"Added: InstitutionName = {ds_copy.InstitutionName}")

# ============================================================
# 6. Working with Sequences
# ============================================================

print("\n" + "=" * 50)
print("WORKING WITH SEQUENCES")
print("=" * 50)

# Create a sequence item
code_item = Dataset()
code_item.CodeValue = "T-D4000"
code_item.CodingSchemeDesignator = "SRT"
code_item.CodeMeaning = "Chest"

# Add sequence to dataset
ds_copy.AnatomicRegionSequence = Sequence([code_item])

print("Anatomic Region Sequence:")
for i, item in enumerate(ds_copy.AnatomicRegionSequence):
    print(f"  Item {i}:")
    for elem in item:
        print(f"    {elem.keyword}: {elem.value}")

# ============================================================
# 7. Date/Time Parsing
# ============================================================

def parse_dicom_date(date_str):
    """Parse DICOM date to Python date."""
    if date_str and len(date_str) >= 8:
        return datetime.strptime(date_str[:8], '%Y%m%d').date()
    return None

def parse_dicom_time(time_str):
    """Parse DICOM time to Python time."""
    if time_str:
        time_str = time_str.split('.')[0]
        if len(time_str) >= 6:
            return datetime.strptime(time_str[:6], '%H%M%S').time()
    return None

print("\n" + "=" * 50)
print("DATE/TIME PARSING")
print("=" * 50)

study_date = parse_dicom_date(ds.StudyDate)
study_time = parse_dicom_time(ds.StudyTime)
birth_date = parse_dicom_date(ds.PatientBirthDate)

print(f"Study Date: {study_date}")
print(f"Study Time: {study_time}")
print(f"Birth Date: {birth_date}")

if study_date and birth_date:
    age = (study_date - birth_date).days // 365
    print(f"Calculated Age: {age} years")

# ============================================================
# 8. Search Tags
# ============================================================

def search_tags(ds, keyword_pattern):
    """Search for tags containing a pattern."""
    matches = []
    keyword_pattern = keyword_pattern.lower()
    for elem in ds:
        if keyword_pattern in elem.keyword.lower():
            matches.append({
                'tag': f"({elem.tag.group:04X},{elem.tag.element:04X})",
                'keyword': elem.keyword,
                'value': elem.value
            })
    return matches

print("\n" + "=" * 50)
print("SEARCHING TAGS")
print("=" * 50)

patient_tags = search_tags(ds, "patient")
print("Tags containing 'patient':")
for t in patient_tags:
    print(f"  {t['tag']} {t['keyword']}: {t['value']}")

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
What we learned:
1. DICOM tags are identified by (Group, Element) pairs
2. Access tags by keyword, tuple, or Tag object
3. Each tag has a Value Representation (VR) defining its type
4. Modify tags by direct assignment or DataElement
5. Sequences are nested datasets
6. Parse dates/times from DICOM format
7. Search tags by keyword pattern
""")
