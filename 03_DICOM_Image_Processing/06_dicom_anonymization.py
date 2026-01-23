"""
LESSON 6: DICOM Anonymization
Biomedical Image Processing - DICOM Module

Topics:
- Why anonymization is important (HIPAA, GDPR, KVKK)
- Which tags contain PHI (Protected Health Information)
- Different anonymization strategies
- Implementing a DICOM anonymizer
"""

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid
import os
import hashlib
from datetime import datetime, timedelta
import random

print("Libraries imported successfully!")

# ============================================================
# 1. PHI Tags in DICOM
# ============================================================

PHI_TAGS = {
    # Patient Information
    (0x0010, 0x0010): 'PatientName',
    (0x0010, 0x0020): 'PatientID',
    (0x0010, 0x0030): 'PatientBirthDate',
    (0x0010, 0x0040): 'PatientSex',
    (0x0010, 0x1010): 'PatientAge',
    (0x0010, 0x1040): 'PatientAddress',
    (0x0010, 0x2154): 'PatientTelephoneNumbers',

    # Study Information
    (0x0008, 0x0020): 'StudyDate',
    (0x0008, 0x0050): 'AccessionNumber',
    (0x0008, 0x0080): 'InstitutionName',
    (0x0008, 0x0090): 'ReferringPhysicianName',
    (0x0008, 0x1010): 'StationName',

    # UIDs
    (0x0020, 0x000D): 'StudyInstanceUID',
    (0x0020, 0x000E): 'SeriesInstanceUID',
    (0x0008, 0x0018): 'SOPInstanceUID',
}

print("PHI TAGS TO ANONYMIZE")
print("=" * 50)
for tag, name in PHI_TAGS.items():
    print(f"({tag[0]:04X},{tag[1]:04X}) - {name}")

# ============================================================
# 2. Create Sample DICOM with PHI
# ============================================================

def create_sample_with_phi():
    """Create a sample DICOM with realistic PHI."""
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

    # Patient PHI
    ds.PatientName = "Yilmaz^Ahmet^Bay"
    ds.PatientID = "TC12345678901"
    ds.PatientBirthDate = "19850315"
    ds.PatientSex = "M"
    ds.PatientAge = "039Y"
    ds.PatientWeight = 78.5
    ds.PatientAddress = "Ataturk Cad. No:123 Ankara"
    ds.PatientTelephoneNumbers = "+90 532 123 4567"

    # Study PHI
    ds.StudyDate = "20240115"
    ds.StudyTime = "143052"
    ds.StudyDescription = "BT Toraks"
    ds.AccessionNumber = "ACC2024011501"
    ds.InstitutionName = "Ankara Universitesi Tip Fakultesi"
    ds.ReferringPhysicianName = "Dr^Mehmet^Demir"

    # UIDs
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.FrameOfReferenceUID = generate_uid()

    # Other metadata
    ds.Modality = "CT"
    ds.SeriesDescription = "Axial 2.5mm"
    ds.Rows = 64
    ds.Columns = 64
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleIntercept = -1024
    ds.RescaleSlope = 1
    ds.WindowCenter = 40
    ds.WindowWidth = 400

    np.random.seed(42)
    pixel_array = np.random.randint(900, 1100, (64, 64), dtype=np.int16)
    ds.PixelData = pixel_array.tobytes()

    return ds

original_ds = create_sample_with_phi()

print("\n" + "=" * 50)
print("ORIGINAL DICOM WITH PHI")
print("=" * 50)
print(f"Patient Name: {original_ds.PatientName}")
print(f"Patient ID: {original_ds.PatientID}")
print(f"Birth Date: {original_ds.PatientBirthDate}")
print(f"Address: {original_ds.PatientAddress}")
print(f"Phone: {original_ds.PatientTelephoneNumbers}")
print(f"Institution: {original_ds.InstitutionName}")

# ============================================================
# 3. DICOM Anonymizer Class
# ============================================================

class DicomAnonymizer:
    """DICOM Anonymization utility with multiple strategies."""

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.date_shift_days = random.randint(-365, 365)
        self.hash_cache = {}
        self.anon_counter = 0

    def _hash_value(self, value, length=8):
        """Create a hash of a value."""
        str_value = str(value)
        if str_value not in self.hash_cache:
            hash_obj = hashlib.sha256(str_value.encode())
            self.hash_cache[str_value] = hash_obj.hexdigest()[:length].upper()
        return self.hash_cache[str_value]

    def _shift_date(self, date_str):
        """Shift a DICOM date by the configured offset."""
        if not date_str or len(date_str) < 8:
            return date_str
        try:
            date_obj = datetime.strptime(date_str[:8], '%Y%m%d')
            shifted = date_obj + timedelta(days=self.date_shift_days)
            return shifted.strftime('%Y%m%d')
        except:
            return date_str

    def _generate_anon_id(self):
        """Generate an anonymous ID."""
        self.anon_counter += 1
        return f"ANON{self.anon_counter:06d}"

    def anonymize(self, ds, strategy='replace'):
        """
        Anonymize a DICOM dataset.

        Parameters:
        -----------
        ds : Dataset
            DICOM dataset to anonymize
        strategy : str
            'remove', 'empty', 'replace', or 'hash'

        Returns:
        --------
        Dataset : Anonymized copy
        """
        anon_ds = ds.copy()

        if strategy == 'remove':
            self._remove_tags(anon_ds)
        elif strategy == 'empty':
            self._empty_tags(anon_ds)
        elif strategy == 'hash':
            self._hash_tags(anon_ds)
        else:
            self._replace_tags(anon_ds)

        self._shift_dates(anon_ds)
        self._regenerate_uids(anon_ds)

        return anon_ds

    def _remove_tags(self, ds):
        """Remove PHI tags."""
        tags = ['PatientName', 'PatientID', 'PatientBirthDate',
                'PatientAddress', 'PatientTelephoneNumbers',
                'ReferringPhysicianName', 'InstitutionName', 'AccessionNumber']
        for tag in tags:
            if hasattr(ds, tag):
                delattr(ds, tag)

    def _empty_tags(self, ds):
        """Set PHI tags to empty."""
        if hasattr(ds, 'PatientName'): ds.PatientName = ''
        if hasattr(ds, 'PatientID'): ds.PatientID = ''
        if hasattr(ds, 'PatientBirthDate'): ds.PatientBirthDate = ''
        if hasattr(ds, 'PatientAddress'): ds.PatientAddress = ''
        if hasattr(ds, 'PatientTelephoneNumbers'): ds.PatientTelephoneNumbers = ''
        if hasattr(ds, 'ReferringPhysicianName'): ds.ReferringPhysicianName = ''
        if hasattr(ds, 'InstitutionName'): ds.InstitutionName = ''

    def _replace_tags(self, ds):
        """Replace PHI tags with dummy values."""
        anon_id = self._generate_anon_id()

        if hasattr(ds, 'PatientName'): ds.PatientName = f"Anonymous^Patient^{anon_id}"
        if hasattr(ds, 'PatientID'): ds.PatientID = anon_id
        if hasattr(ds, 'PatientBirthDate'): ds.PatientBirthDate = '19000101'
        if hasattr(ds, 'PatientAge'): ds.PatientAge = '000Y'
        if hasattr(ds, 'PatientAddress'): ds.PatientAddress = 'ANONYMIZED'
        if hasattr(ds, 'PatientTelephoneNumbers'): ds.PatientTelephoneNumbers = ''
        if hasattr(ds, 'ReferringPhysicianName'): ds.ReferringPhysicianName = 'ANONYMIZED'
        if hasattr(ds, 'InstitutionName'): ds.InstitutionName = 'ANONYMIZED'
        if hasattr(ds, 'AccessionNumber'): ds.AccessionNumber = anon_id

    def _hash_tags(self, ds):
        """Replace PHI tags with hashed values."""
        if hasattr(ds, 'PatientName'):
            hash_val = self._hash_value(ds.PatientName)
            ds.PatientName = f"HASH^{hash_val}"
        if hasattr(ds, 'PatientID'):
            ds.PatientID = self._hash_value(ds.PatientID, 12)
        if hasattr(ds, 'PatientBirthDate'): ds.PatientBirthDate = '19000101'
        if hasattr(ds, 'PatientAddress'): ds.PatientAddress = ''
        if hasattr(ds, 'PatientTelephoneNumbers'): ds.PatientTelephoneNumbers = ''
        if hasattr(ds, 'InstitutionName'):
            ds.InstitutionName = self._hash_value(ds.InstitutionName)

    def _shift_dates(self, ds):
        """Shift all dates."""
        date_tags = ['StudyDate', 'SeriesDate', 'AcquisitionDate']
        for tag in date_tags:
            if hasattr(ds, tag):
                setattr(ds, tag, self._shift_date(getattr(ds, tag)))

    def _regenerate_uids(self, ds):
        """Regenerate all UIDs."""
        if hasattr(ds, 'StudyInstanceUID'):
            ds.StudyInstanceUID = generate_uid()
        if hasattr(ds, 'SeriesInstanceUID'):
            ds.SeriesInstanceUID = generate_uid()
        if hasattr(ds, 'SOPInstanceUID'):
            ds.SOPInstanceUID = generate_uid()
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        if hasattr(ds, 'FrameOfReferenceUID'):
            ds.FrameOfReferenceUID = generate_uid()

# ============================================================
# 4. Test Different Strategies
# ============================================================

def compare_anonymization(original, strategy):
    """Compare original and anonymized datasets."""
    anonymizer = DicomAnonymizer(seed=42)
    anon = anonymizer.anonymize(original, strategy=strategy)

    print(f"\n{'='*60}")
    print(f"STRATEGY: {strategy.upper()}")
    print(f"{'='*60}")

    tags = ['PatientName', 'PatientID', 'PatientBirthDate',
            'StudyDate', 'InstitutionName']

    print(f"{'Tag':<20} {'Original':<25} {'Anonymized'}")
    print("-" * 70)

    for tag in tags:
        orig_val = str(getattr(original, tag, 'N/A'))[:24]
        anon_val = str(getattr(anon, tag, '[REMOVED]'))[:24] if hasattr(anon, tag) else '[REMOVED]'
        print(f"{tag:<20} {orig_val:<25} {anon_val}")

    return anon

# Test all strategies
for strategy in ['remove', 'empty', 'replace', 'hash']:
    compare_anonymization(original_ds, strategy)

# ============================================================
# 5. Verification
# ============================================================

def verify_anonymization(ds):
    """Verify that a DICOM has been properly anonymized."""
    warnings = []

    checks = [
        ('PatientName', lambda v: v and 'ANON' not in str(v).upper() and 'HASH' not in str(v).upper()),
        ('PatientID', lambda v: v and len(str(v)) > 12 and not str(v).startswith('ANON')),
        ('PatientBirthDate', lambda v: v and v != '19000101' and v != ''),
        ('PatientAddress', lambda v: v and v != '' and v != 'ANONYMIZED'),
    ]

    for tag, check_func in checks:
        if hasattr(ds, tag):
            value = getattr(ds, tag)
            if check_func(value):
                warnings.append(f"Potential PHI in {tag}: {value}")

    is_safe = len(warnings) == 0

    print("\nANONYMIZATION VERIFICATION")
    print("=" * 40)
    if is_safe:
        print("[PASS] No obvious PHI detected")
    else:
        print("[WARN] Potential PHI found:")
        for w in warnings:
            print(f"  - {w}")

    return is_safe, warnings

print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

print("\nChecking ORIGINAL dataset:")
verify_anonymization(original_ds)

print("\nChecking ANONYMIZED dataset:")
anonymizer = DicomAnonymizer(seed=42)
anon_ds = anonymizer.anonymize(original_ds, strategy='replace')
verify_anonymization(anon_ds)

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
What we learned:
1. PHI must be removed/modified before sharing DICOM
2. Multiple strategies: remove, empty, replace, hash
3. Date shifting preserves temporal relationships
4. UID regeneration prevents linking
5. Verification is essential - always check your work
6. HIPAA, GDPR, KVKK compliance is mandatory
""")
