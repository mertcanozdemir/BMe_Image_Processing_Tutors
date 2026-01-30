"""
LESSON 6: Histogram Equalization and Histogram Matching
=======================================================
Biomedical Image Processing Techniques

In this lesson:
- Histogram computation and visualization
- Normalized histogram (PDF) and CDF
- Histogram equalization algorithm
- Histogram matching (specification)
- Medical imaging applications
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. HISTOGRAM COMPUTATION
# =============================================================================
# A histogram shows the distribution of pixel intensities in an image.
# h(r_k) = n_k
# Where:
#   r_k = intensity level k
#   n_k = number of pixels with intensity r_k

print("="*50)
print("HISTOGRAM COMPUTATION")
print("="*50)
print("""
Histogram: h(r_k) = n_k

Where:
- r_k = intensity level k (0 to 255 for 8-bit)
- n_k = number of pixels with intensity r_k
""")

def compute_histogram(image):
    """
    Compute histogram of a grayscale image (manual method).

    Parameters:
    - image: input grayscale image (uint8)

    Returns:
    - histogram: array of 256 values
    """
    histogram = np.zeros(256, dtype=np.int32)

    for pixel_value in image.ravel():
        histogram[pixel_value] += 1

    return histogram

def compute_histogram_fast(image):
    """
    Compute histogram using numpy (fast method).

    Parameters:
    - image: input grayscale image (uint8)

    Returns:
    - histogram: array of 256 values
    """
    return np.bincount(image.ravel(), minlength=256)

# Create sample images with different distributions
np.random.seed(42)

# Dark image (low intensity)
dark_image = np.random.randint(0, 80, (200, 200), dtype=np.uint8)

# Bright image (high intensity)
bright_image = np.random.randint(180, 255, (200, 200), dtype=np.uint8)

# Low contrast (narrow range)
low_contrast = np.random.randint(100, 150, (200, 200), dtype=np.uint8)

# Normal distribution
normal_image = np.clip(
    np.random.normal(128, 40, (200, 200)), 0, 255
).astype(np.uint8)

# Visualize images and their histograms
images = [dark_image, bright_image, low_contrast, normal_image]
titles = ['Dark Image', 'Bright Image', 'Low Contrast', 'Normal Distribution']

plt.figure(figsize=(16, 8))

for i, (img, title) in enumerate(zip(images, titles)):
    # Image
    plt.subplot(2, 4, i+1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')

    # Histogram
    plt.subplot(2, 4, i+5)
    hist = compute_histogram_fast(img)
    plt.bar(range(256), hist, color='gray', width=1)
    plt.xlim([0, 255])
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.title('Histogram')

plt.suptitle('Images and Their Histograms', fontsize=14)
plt.tight_layout()
plt.savefig('01_histograms.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 2. NORMALIZED HISTOGRAM (PDF) AND CDF
# =============================================================================
# PDF: p(r_k) = n_k / N   (N = total number of pixels)
# CDF: CDF(r_k) = sum of p(r_j) for j = 0 to k

print("\n" + "="*50)
print("NORMALIZED HISTOGRAM (PDF) AND CDF")
print("="*50)
print("""
PDF:  p(r_k) = n_k / N   (probability of intensity r_k)
CDF:  CDF(r_k) = sum(p(r_j)) for j = 0..k

- PDF sums to 1.0
- CDF is monotonically increasing from 0 to 1
""")

def normalized_histogram(image):
    """
    Compute normalized histogram (probability distribution function).

    Parameters:
    - image: input grayscale image (uint8)

    Returns:
    - pdf: probability for each intensity level (256 values)
    """
    hist = compute_histogram_fast(image)
    total_pixels = image.size
    pdf = hist / total_pixels
    return pdf

def compute_cdf(image):
    """
    Compute cumulative distribution function.

    Parameters:
    - image: input grayscale image (uint8)

    Returns:
    - cdf: cumulative distribution for each intensity (256 values)
    """
    pdf = normalized_histogram(image)
    cdf = np.cumsum(pdf)
    return cdf

# Visualize PDF and CDF for different images
plt.figure(figsize=(15, 5))

for i, (img, title) in enumerate(zip(
        [dark_image, low_contrast, normal_image],
        ['Dark Image', 'Low Contrast', 'Normal'])):
    plt.subplot(1, 3, i+1)

    pdf = normalized_histogram(img)
    cdf = compute_cdf(img)

    plt.bar(range(256), pdf, color='steelblue', width=1, alpha=0.7, label='PDF')
    plt.plot(range(256), cdf, 'r-', linewidth=2, label='CDF')

    plt.xlabel('Intensity')
    plt.ylabel('Probability / CDF')
    plt.title(title)
    plt.legend()
    plt.xlim([0, 255])
    plt.ylim([0, 1.1])

plt.suptitle('PDF and CDF Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('02_pdf_cdf.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 3. HISTOGRAM EQUALIZATION
# =============================================================================
# Goal: Transform image so that the output histogram is approximately uniform.
#
# Algorithm:
#   1. Compute histogram h(r_k)
#   2. Compute normalized histogram (PDF) p(r_k)
#   3. Compute CDF(r_k)
#   4. Transform: s_k = (L-1) * CDF(r_k)   where L=256 for 8-bit images

print("\n" + "="*50)
print("HISTOGRAM EQUALIZATION")
print("="*50)
print("""
Goal: Transform image to have approximately uniform histogram.

Algorithm:
  1. Compute histogram h(r_k)
  2. Compute PDF p(r_k) = h(r_k) / N
  3. Compute CDF(r_k) = cumulative sum of p
  4. Map: s_k = round((L-1) * CDF(r_k))

Where L = 256 for 8-bit images.
""")

def histogram_equalization(image):
    """
    Perform histogram equalization on a grayscale image.

    Parameters:
    - image: input grayscale image (uint8)

    Returns:
    - equalized: output image with equalized histogram (uint8)
    - lookup_table: the transformation mapping (256 values)
    """
    # Step 1: Compute histogram
    hist = compute_histogram_fast(image)

    # Step 2: Compute PDF
    pdf = hist / image.size

    # Step 3: Compute CDF
    cdf = np.cumsum(pdf)

    # Step 4: Create lookup table
    # s_k = (L-1) * CDF(r_k)
    lookup_table = np.round(255 * cdf).astype(np.uint8)

    # Step 5: Apply transformation
    equalized = lookup_table[image]

    return equalized, lookup_table

# Apply histogram equalization to different images
test_images = [dark_image, bright_image, low_contrast]
test_titles = ['Dark Image', 'Bright Image', 'Low Contrast']

plt.figure(figsize=(16, 12))

for i, (img, title) in enumerate(zip(test_images, test_titles)):
    equalized, lut = histogram_equalization(img)

    # Original image
    plt.subplot(3, 4, i*4 + 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Original: {title}')
    plt.axis('off')

    # Original histogram
    plt.subplot(3, 4, i*4 + 2)
    plt.hist(img.ravel(), bins=256, range=(0, 256), color='steelblue', alpha=0.7)
    plt.xlim([0, 255])
    plt.title('Original Histogram')

    # Equalized image
    plt.subplot(3, 4, i*4 + 3)
    plt.imshow(equalized, cmap='gray', vmin=0, vmax=255)
    plt.title('Equalized')
    plt.axis('off')

    # Equalized histogram
    plt.subplot(3, 4, i*4 + 4)
    plt.hist(equalized.ravel(), bins=256, range=(0, 256), color='coral', alpha=0.7)
    plt.xlim([0, 255])
    plt.title('Equalized Histogram')

plt.suptitle('Histogram Equalization Results', fontsize=14)
plt.tight_layout()
plt.savefig('03_histogram_equalization.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 4. UNDERSTANDING THE TRANSFORMATION FUNCTION
# =============================================================================

print("\n" + "="*50)
print("TRANSFORMATION FUNCTIONS")
print("="*50)

plt.figure(figsize=(15, 5))

for i, (img, title) in enumerate(zip(test_images, test_titles)):
    _, lut = histogram_equalization(img)

    plt.subplot(1, 3, i+1)
    plt.plot(range(256), lut, 'b-', linewidth=2)
    plt.plot([0, 255], [0, 255], 'k--', alpha=0.5, label='Identity')
    plt.xlabel('Input Intensity')
    plt.ylabel('Output Intensity')
    plt.title(f'Transformation: {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 255])
    plt.ylim([0, 255])

plt.suptitle('Histogram Equalization Transformation Functions', fontsize=14)
plt.tight_layout()
plt.savefig('04_transformation_functions.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 5. STEP-BY-STEP DEMONSTRATION
# =============================================================================

print("\n" + "="*50)
print("STEP-BY-STEP DEMONSTRATION")
print("="*50)

# Detailed step-by-step for dark image
img = dark_image

# Step 1: Histogram
hist = compute_histogram_fast(img)

# Step 2: PDF
pdf = hist / img.size

# Step 3: CDF
cdf = np.cumsum(pdf)

# Step 4: Lookup table
lut = np.round(255 * cdf).astype(np.uint8)

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.bar(range(256), hist, color='steelblue', width=1)
plt.title('Step 1: Histogram h(r)')
plt.xlabel('Intensity')
plt.xlim([0, 255])

plt.subplot(1, 4, 2)
plt.bar(range(256), pdf, color='coral', width=1)
plt.title('Step 2: PDF p(r)')
plt.xlabel('Intensity')
plt.xlim([0, 255])

plt.subplot(1, 4, 3)
plt.plot(range(256), cdf, 'g-', linewidth=2)
plt.title('Step 3: CDF')
plt.xlabel('Intensity')
plt.ylabel('Cumulative Probability')
plt.xlim([0, 255])
plt.ylim([0, 1])
plt.grid(True, alpha=0.3)

plt.subplot(1, 4, 4)
plt.plot(range(256), lut, 'r-', linewidth=2)
plt.title('Step 4: Lookup Table\ns = 255 * CDF(r)')
plt.xlabel('Input')
plt.ylabel('Output')
plt.xlim([0, 255])
plt.ylim([0, 255])
plt.grid(True, alpha=0.3)

plt.suptitle('Histogram Equalization: Step by Step (Dark Image)', fontsize=14)
plt.tight_layout()
plt.savefig('05_step_by_step.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 6. HISTOGRAM MATCHING (SPECIFICATION)
# =============================================================================
# Goal: Transform image so that its histogram matches a SPECIFIED distribution.
# Unlike equalization (which targets uniform), matching targets ANY distribution.
#
# Algorithm:
#   1. Compute CDF of source image: CDF_source
#   2. Compute CDF of reference image (or target distribution): CDF_ref
#   3. For each intensity s in source:
#      a. Find CDF_source(s)
#      b. Find intensity r in reference where CDF_ref(r) is closest to CDF_source(s)
#      c. Map s -> r

print("\n" + "="*50)
print("HISTOGRAM MATCHING (SPECIFICATION)")
print("="*50)
print("""
Goal: Transform image so its histogram matches a TARGET distribution.

Unlike equalization (uniform target), matching uses ANY target.

Algorithm:
  1. Compute CDF of source image: CDF_src
  2. Compute CDF of reference image: CDF_ref
  3. For each source intensity s:
     - Find r where CDF_ref(r) is closest to CDF_src(s)
     - Map: s -> r
""")

def histogram_matching(source, reference):
    """
    Match the histogram of a source image to a reference image.

    Parameters:
    - source: input image to be transformed (uint8)
    - reference: target image whose histogram we want to match (uint8)

    Returns:
    - matched: output image with matched histogram (uint8)
    - lookup_table: the transformation mapping (256 values)
    """
    # Step 1: Compute CDF of source
    src_hist = compute_histogram_fast(source)
    src_cdf = np.cumsum(src_hist / source.size)

    # Step 2: Compute CDF of reference
    ref_hist = compute_histogram_fast(reference)
    ref_cdf = np.cumsum(ref_hist / reference.size)

    # Step 3: Create lookup table
    # For each source intensity, find the closest matching reference intensity
    lookup_table = np.zeros(256, dtype=np.uint8)

    for src_val in range(256):
        # Find the reference intensity whose CDF value is closest
        diff = np.abs(ref_cdf - src_cdf[src_val])
        lookup_table[src_val] = np.argmin(diff)

    # Step 4: Apply transformation
    matched = lookup_table[source]

    return matched, lookup_table

# --- Example 1: Match dark image to normal distribution ---
print("\nExample 1: Matching dark image to normal distribution")

matched_dark, lut_dark = histogram_matching(dark_image, normal_image)

plt.figure(figsize=(18, 10))

# Source
plt.subplot(2, 3, 1)
plt.imshow(dark_image, cmap='gray', vmin=0, vmax=255)
plt.title('Source: Dark Image')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.hist(dark_image.ravel(), bins=256, range=(0, 256), color='steelblue', alpha=0.7)
plt.xlim([0, 255])
plt.title('Source Histogram')
plt.xlabel('Intensity')

# Reference
plt.subplot(2, 3, 2)
plt.imshow(normal_image, cmap='gray', vmin=0, vmax=255)
plt.title('Reference: Normal Distribution')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.hist(normal_image.ravel(), bins=256, range=(0, 256), color='forestgreen', alpha=0.7)
plt.xlim([0, 255])
plt.title('Reference Histogram')
plt.xlabel('Intensity')

# Result
plt.subplot(2, 3, 3)
plt.imshow(matched_dark, cmap='gray', vmin=0, vmax=255)
plt.title('Result: Matched Image')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.hist(matched_dark.ravel(), bins=256, range=(0, 256), color='coral', alpha=0.7)
plt.xlim([0, 255])
plt.title('Matched Histogram')
plt.xlabel('Intensity')

plt.suptitle('Histogram Matching: Dark Image -> Normal Distribution', fontsize=14)
plt.tight_layout()
plt.savefig('06_histogram_matching_example1.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Example 2: Match low contrast image to bright image ---
print("Example 2: Matching low contrast image to bright image")

matched_low, lut_low = histogram_matching(low_contrast, bright_image)

plt.figure(figsize=(18, 10))

# Source
plt.subplot(2, 3, 1)
plt.imshow(low_contrast, cmap='gray', vmin=0, vmax=255)
plt.title('Source: Low Contrast')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.hist(low_contrast.ravel(), bins=256, range=(0, 256), color='steelblue', alpha=0.7)
plt.xlim([0, 255])
plt.title('Source Histogram')
plt.xlabel('Intensity')

# Reference
plt.subplot(2, 3, 2)
plt.imshow(bright_image, cmap='gray', vmin=0, vmax=255)
plt.title('Reference: Bright Image')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.hist(bright_image.ravel(), bins=256, range=(0, 256), color='forestgreen', alpha=0.7)
plt.xlim([0, 255])
plt.title('Reference Histogram')
plt.xlabel('Intensity')

# Result
plt.subplot(2, 3, 3)
plt.imshow(matched_low, cmap='gray', vmin=0, vmax=255)
plt.title('Result: Matched Image')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.hist(matched_low.ravel(), bins=256, range=(0, 256), color='coral', alpha=0.7)
plt.xlim([0, 255])
plt.title('Matched Histogram')
plt.xlabel('Intensity')

plt.suptitle('Histogram Matching: Low Contrast -> Bright Image', fontsize=14)
plt.tight_layout()
plt.savefig('07_histogram_matching_example2.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 7. MATCHING TRANSFORMATION VISUALIZATION
# =============================================================================

print("\n" + "="*50)
print("MATCHING TRANSFORMATION FUNCTIONS")
print("="*50)

plt.figure(figsize=(15, 5))

# Transformation for dark -> normal
plt.subplot(1, 3, 1)
plt.plot(range(256), lut_dark, 'b-', linewidth=2)
plt.plot([0, 255], [0, 255], 'k--', alpha=0.5, label='Identity')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')
plt.title('Dark -> Normal')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 255])
plt.ylim([0, 255])

# Transformation for low contrast -> bright
plt.subplot(1, 3, 2)
plt.plot(range(256), lut_low, 'r-', linewidth=2)
plt.plot([0, 255], [0, 255], 'k--', alpha=0.5, label='Identity')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')
plt.title('Low Contrast -> Bright')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 255])
plt.ylim([0, 255])

# CDF comparison
plt.subplot(1, 3, 3)
cdf_dark = compute_cdf(dark_image)
cdf_normal = compute_cdf(normal_image)
cdf_matched = compute_cdf(matched_dark)
plt.plot(range(256), cdf_dark, 'b-', linewidth=2, label='Source (Dark)')
plt.plot(range(256), cdf_normal, 'g-', linewidth=2, label='Reference (Normal)')
plt.plot(range(256), cdf_matched, 'r--', linewidth=2, label='Matched Result')
plt.xlabel('Intensity')
plt.ylabel('CDF')
plt.title('CDF Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 255])
plt.ylim([0, 1.05])

plt.suptitle('Histogram Matching: Transformations and CDF', fontsize=14)
plt.tight_layout()
plt.savefig('08_matching_transformations.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 8. MATCHING TO A CUSTOM DISTRIBUTION
# =============================================================================
# You can also match to a synthetic/desired distribution instead of a real image.

print("\n" + "="*50)
print("MATCHING TO CUSTOM DISTRIBUTIONS")
print("="*50)
print("""
Histogram matching is not limited to real reference images.
You can specify ANY target distribution:
  - Gaussian (normal)
  - Bimodal
  - Exponential
  - Any custom shape
""")

def match_to_distribution(source, target_pdf):
    """
    Match the histogram of a source image to a target PDF.

    Parameters:
    - source: input image (uint8)
    - target_pdf: desired probability distribution (256 values, sums to 1)

    Returns:
    - matched: output image (uint8)
    """
    # Source CDF
    src_hist = compute_histogram_fast(source)
    src_cdf = np.cumsum(src_hist / source.size)

    # Target CDF
    target_cdf = np.cumsum(target_pdf)

    # Create lookup table
    lookup_table = np.zeros(256, dtype=np.uint8)
    for s in range(256):
        diff = np.abs(target_cdf - src_cdf[s])
        lookup_table[s] = np.argmin(diff)

    return lookup_table[source]

# Create custom target distributions
x = np.arange(256)

# Gaussian distribution centered at 128
gaussian_pdf = np.exp(-0.5 * ((x - 128) / 30) ** 2)
gaussian_pdf = gaussian_pdf / gaussian_pdf.sum()

# Bimodal distribution (two peaks)
bimodal_pdf = (np.exp(-0.5 * ((x - 80) / 20) ** 2) +
               np.exp(-0.5 * ((x - 180) / 20) ** 2))
bimodal_pdf = bimodal_pdf / bimodal_pdf.sum()

# Exponential distribution
exp_pdf = np.exp(-x / 50.0)
exp_pdf = exp_pdf / exp_pdf.sum()

# Apply matching
matched_gauss = match_to_distribution(low_contrast, gaussian_pdf)
matched_bimodal = match_to_distribution(low_contrast, bimodal_pdf)
matched_exp = match_to_distribution(low_contrast, exp_pdf)

plt.figure(figsize=(16, 12))

# Original
plt.subplot(3, 4, 1)
plt.imshow(low_contrast, cmap='gray', vmin=0, vmax=255)
plt.title('Source: Low Contrast')
plt.axis('off')

plt.subplot(3, 4, 2)
plt.hist(low_contrast.ravel(), bins=256, range=(0, 256), color='steelblue', alpha=0.7)
plt.xlim([0, 255])
plt.title('Source Histogram')

# Gaussian target
plt.subplot(3, 4, 5)
plt.imshow(matched_gauss, cmap='gray', vmin=0, vmax=255)
plt.title('Matched: Gaussian')
plt.axis('off')

plt.subplot(3, 4, 6)
plt.hist(matched_gauss.ravel(), bins=256, range=(0, 256), color='coral', alpha=0.7)
plt.xlim([0, 255])
plt.title('Gaussian Target Result')

plt.subplot(3, 4, 3)
plt.bar(x, gaussian_pdf, color='forestgreen', width=1)
plt.xlim([0, 255])
plt.title('Target: Gaussian PDF')

# Bimodal target
plt.subplot(3, 4, 9)
plt.imshow(matched_bimodal, cmap='gray', vmin=0, vmax=255)
plt.title('Matched: Bimodal')
plt.axis('off')

plt.subplot(3, 4, 10)
plt.hist(matched_bimodal.ravel(), bins=256, range=(0, 256), color='coral', alpha=0.7)
plt.xlim([0, 255])
plt.title('Bimodal Target Result')

plt.subplot(3, 4, 7)
plt.bar(x, bimodal_pdf, color='forestgreen', width=1)
plt.xlim([0, 255])
plt.title('Target: Bimodal PDF')

# Exponential target
plt.subplot(3, 4, 4)
plt.bar(x, exp_pdf, color='forestgreen', width=1)
plt.xlim([0, 255])
plt.title('Target: Exponential PDF')

plt.subplot(3, 4, 8)
plt.imshow(matched_exp, cmap='gray', vmin=0, vmax=255)
plt.title('Matched: Exponential')
plt.axis('off')

plt.subplot(3, 4, 12)
plt.hist(matched_exp.ravel(), bins=256, range=(0, 256), color='coral', alpha=0.7)
plt.xlim([0, 255])
plt.title('Exponential Target Result')

plt.suptitle('Histogram Matching to Custom Distributions', fontsize=14)
plt.tight_layout()
plt.savefig('09_custom_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 9. MEDICAL IMAGING APPLICATION
# =============================================================================
# Histogram equalization and matching are widely used in medical imaging to:
# - Enhance contrast in X-ray, CT, and MRI images
# - Standardize appearance across different scanners
# - Pre-process images for further analysis

print("\n" + "="*50)
print("MEDICAL IMAGING APPLICATION")
print("="*50)

# Simulate a low-contrast X-ray image
np.random.seed(456)

xray = np.random.randint(60, 100, (256, 256), dtype=np.uint8)

# Add "bone" structures
xray[50:200, 100:120] = np.random.randint(100, 130, (150, 20), dtype=np.uint8)  # Spine
xray[80:120, 60:180] = np.random.randint(90, 120, (40, 120), dtype=np.uint8)    # Ribs
xray[140:180, 60:180] = np.random.randint(90, 120, (40, 120), dtype=np.uint8)   # Ribs

# Add darker areas (lungs)
xray[70:190, 40:90] = np.random.randint(40, 70, (120, 50), dtype=np.uint8)
xray[70:190, 140:190] = np.random.randint(40, 70, (120, 50), dtype=np.uint8)

# Smooth for realism
from scipy.ndimage import gaussian_filter
xray = gaussian_filter(xray, sigma=2).astype(np.uint8)

# Apply histogram equalization
xray_equalized, _ = histogram_equalization(xray)

# Create a "well-exposed" reference for matching
np.random.seed(789)
reference_xray = np.random.randint(30, 220, (256, 256), dtype=np.uint8)
reference_xray[50:200, 100:120] = np.random.randint(150, 200, (150, 20), dtype=np.uint8)
reference_xray[80:120, 60:180] = np.random.randint(130, 180, (40, 120), dtype=np.uint8)
reference_xray[140:180, 60:180] = np.random.randint(130, 180, (40, 120), dtype=np.uint8)
reference_xray[70:190, 40:90] = np.random.randint(20, 60, (120, 50), dtype=np.uint8)
reference_xray[70:190, 140:190] = np.random.randint(20, 60, (120, 50), dtype=np.uint8)
reference_xray = gaussian_filter(reference_xray, sigma=2).astype(np.uint8)

# Apply histogram matching
xray_matched, _ = histogram_matching(xray, reference_xray)

plt.figure(figsize=(16, 10))

# Original X-ray
plt.subplot(2, 4, 1)
plt.imshow(xray, cmap='gray')
plt.title('Original X-Ray\n(Low Contrast)')
plt.axis('off')
plt.colorbar(fraction=0.046)

plt.subplot(2, 4, 5)
plt.hist(xray.ravel(), bins=256, range=(0, 256), color='steelblue', alpha=0.7)
plt.xlim([0, 255])
plt.title('Original Histogram')

# Equalized
plt.subplot(2, 4, 2)
plt.imshow(xray_equalized, cmap='gray')
plt.title('Histogram Equalization\n(Uniform Target)')
plt.axis('off')
plt.colorbar(fraction=0.046)

plt.subplot(2, 4, 6)
plt.hist(xray_equalized.ravel(), bins=256, range=(0, 256), color='coral', alpha=0.7)
plt.xlim([0, 255])
plt.title('Equalized Histogram')

# Reference
plt.subplot(2, 4, 3)
plt.imshow(reference_xray, cmap='gray')
plt.title('Reference X-Ray\n(Well-Exposed)')
plt.axis('off')
plt.colorbar(fraction=0.046)

plt.subplot(2, 4, 7)
plt.hist(reference_xray.ravel(), bins=256, range=(0, 256), color='forestgreen', alpha=0.7)
plt.xlim([0, 255])
plt.title('Reference Histogram')

# Matched
plt.subplot(2, 4, 4)
plt.imshow(xray_matched, cmap='gray')
plt.title('Histogram Matching\n(Reference Target)')
plt.axis('off')
plt.colorbar(fraction=0.046)

plt.subplot(2, 4, 8)
plt.hist(xray_matched.ravel(), bins=256, range=(0, 256), color='orchid', alpha=0.7)
plt.xlim([0, 255])
plt.title('Matched Histogram')

plt.suptitle('Medical Imaging: Equalization vs Matching', fontsize=14)
plt.tight_layout()
plt.savefig('10_medical_application.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 10. EQUALIZATION VS MATCHING COMPARISON
# =============================================================================

print("\n" + "="*50)
print("EQUALIZATION VS MATCHING: COMPARISON")
print("="*50)
print("""
Histogram Equalization:
  - Target: UNIFORM distribution
  - No reference image needed
  - Always maximizes contrast
  - Can over-enhance some images

Histogram Matching (Specification):
  - Target: ANY specified distribution
  - Requires a reference image or target PDF
  - More control over output appearance
  - Useful for standardizing images from different sources
""")

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.imshow(dark_image, cmap='gray', vmin=0, vmax=255)
plt.title('Original (Dark)')
plt.axis('off')

plt.subplot(1, 3, 2)
equalized, _ = histogram_equalization(dark_image)
plt.imshow(equalized, cmap='gray', vmin=0, vmax=255)
plt.title('Equalized\n(Uniform Target)')
plt.axis('off')

plt.subplot(1, 3, 3)
matched, _ = histogram_matching(dark_image, normal_image)
plt.imshow(matched, cmap='gray', vmin=0, vmax=255)
plt.title('Matched\n(Gaussian Target)')
plt.axis('off')

plt.suptitle('Equalization vs Matching: Visual Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('11_equalization_vs_matching.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("LESSON 6 COMPLETED!")
print("="*50)
print("""
What we learned:
1. Histogram: distribution of pixel intensities h(r_k) = n_k
2. PDF: normalized histogram p(r_k) = n_k / N
3. CDF: cumulative distribution CDF(r_k) = sum(p(r_j))
4. Histogram Equalization:
   - Targets uniform distribution
   - Formula: s = 255 * CDF(r)
   - Automatically enhances contrast
5. Histogram Matching (Specification):
   - Targets ANY specified distribution
   - Maps via CDF of source and reference
   - More control than equalization
6. Medical Imaging:
   - Enhance low-contrast medical images
   - Standardize images across scanners
   - Pre-processing for automated analysis
""")
