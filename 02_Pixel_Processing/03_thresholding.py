"""
LESSON 5: Thresholding (Image Binarization)
===========================================
Biomedical Image Processing Techniques

In this lesson:
- Simple (global) thresholding
- Binary and inverse binary thresholding
- Truncate, to-zero thresholding
- Practical applications in biomedical imaging
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. WHAT IS THRESHOLDING?
# =============================================================================
# Thresholding converts a grayscale image to a binary image
# Based on a threshold value T:
#   - Pixels >= T become WHITE (255)
#   - Pixels < T become BLACK (0)

print("="*50)
print("THRESHOLDING (BINARIZATION)")
print("="*50)
print("""
Thresholding: Convert grayscale to binary

Simple threshold formula:
    if f(x,y) >= T:
        g(x,y) = 255 (white)
    else:
        g(x,y) = 0 (black)

Used for: Segmentation, object detection, cell counting
""")

# Create a sample image with objects
np.random.seed(42)
image = np.zeros((200, 200), dtype=np.uint8)
# Add some "objects" (bright regions)
image[30:70, 30:70] = 180      # Square
image[50:90, 120:180] = 200    # Rectangle
image[130:180, 40:100] = 160   # Another rectangle
image[120:170, 130:190] = 220  # Bright square
# Add noise
image = image + np.random.randint(0, 40, (200, 200), dtype=np.uint8)

# =============================================================================
# 2. SIMPLE (GLOBAL) THRESHOLDING
# =============================================================================

def threshold_binary(image, thresh):
    """
    Apply binary thresholding.
    Pixels >= thresh become 255, otherwise 0.
    """
    result = np.zeros_like(image)
    result[image >= thresh] = 255
    return result

def threshold_binary_inv(image, thresh):
    """
    Apply inverse binary thresholding.
    Pixels >= thresh become 0, otherwise 255.
    """
    result = np.ones_like(image) * 255
    result[image >= thresh] = 0
    return result

# Apply thresholding with different values
thresholds = [80, 120, 160, 200]

plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.colorbar()

for i, T in enumerate(thresholds):
    plt.subplot(2, 3, i+2)
    binary = threshold_binary(image, T)
    plt.imshow(binary, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Threshold T = {T}')

plt.subplot(2, 3, 6)
# Show histogram
plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
for T in thresholds:
    plt.axvline(x=T, color='r', linestyle='--', label=f'T={T}')
plt.xlabel('Pixel Value')
plt.ylabel('Count')
plt.title('Histogram with Thresholds')

plt.suptitle('Binary Thresholding: g = 255 if f >= T, else 0', fontsize=14)
plt.tight_layout()
plt.savefig('01_binary_threshold.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 3. DIFFERENT THRESHOLDING TYPES
# =============================================================================

print("\n" + "="*50)
print("THRESHOLDING TYPES")
print("="*50)

def threshold_trunc(image, thresh):
    """Truncate: values above thresh become thresh."""
    result = image.copy()
    result[result > thresh] = thresh
    return result

def threshold_tozero(image, thresh):
    """To-zero: values below thresh become 0."""
    result = image.copy()
    result[result < thresh] = 0
    return result

def threshold_tozero_inv(image, thresh):
    """To-zero inverse: values above thresh become 0."""
    result = image.copy()
    result[result >= thresh] = 0
    return result

T = 128

# Create gradient for demonstration
gradient = np.tile(np.linspace(0, 255, 256), (100, 1)).astype(np.uint8)

results = {
    'Original': gradient,
    'Binary': threshold_binary(gradient, T),
    'Binary Inverse': threshold_binary_inv(gradient, T),
    'Truncate': threshold_trunc(gradient, T),
    'To Zero': threshold_tozero(gradient, T),
    'To Zero Inverse': threshold_tozero_inv(gradient, T)
}

plt.figure(figsize=(15, 10))

for i, (name, img) in enumerate(results.items()):
    plt.subplot(3, 2, i+1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(name)
    plt.colorbar()

plt.suptitle(f'Different Thresholding Types (T = {T})', fontsize=14)
plt.tight_layout()
plt.savefig('02_threshold_types.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 4. THRESHOLD TRANSFORMATION FUNCTIONS
# =============================================================================

print("\n" + "="*50)
print("TRANSFORMATION FUNCTIONS")
print("="*50)

x = np.arange(0, 256)
T = 128

plt.figure(figsize=(15, 5))

# Binary
plt.subplot(1, 3, 1)
y_binary = np.where(x >= T, 255, 0)
plt.plot(x, y_binary, 'b-', linewidth=2)
plt.axvline(x=T, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Binary Threshold')
plt.grid(True, alpha=0.3)
plt.xlim([0, 255])
plt.ylim([-10, 265])

# Truncate
plt.subplot(1, 3, 2)
y_trunc = np.where(x > T, T, x)
plt.plot(x, y_trunc, 'g-', linewidth=2)
plt.axvline(x=T, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Truncate')
plt.grid(True, alpha=0.3)
plt.xlim([0, 255])
plt.ylim([-10, 265])

# To Zero
plt.subplot(1, 3, 3)
y_tozero = np.where(x < T, 0, x)
plt.plot(x, y_tozero, 'm-', linewidth=2)
plt.axvline(x=T, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('To Zero')
plt.grid(True, alpha=0.3)
plt.xlim([0, 255])
plt.ylim([-10, 265])

plt.suptitle(f'Threshold Transformation Functions (T = {T})', fontsize=14)
plt.tight_layout()
plt.savefig('03_transform_functions.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 5. BIOMEDICAL APPLICATION: CELL DETECTION
# =============================================================================

print("\n" + "="*50)
print("BIOMEDICAL APPLICATION: CELL DETECTION")
print("="*50)

# Simulate a microscopy image with cells
np.random.seed(123)
cell_image = np.random.randint(20, 60, (200, 200), dtype=np.uint8)

# Add "cells" as bright circular regions
def add_cell(image, cx, cy, radius, intensity):
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    image[mask] = intensity

# Add several cells
cells = [(40, 50, 15, 180), (120, 40, 20, 200), (80, 120, 18, 190),
         (160, 100, 12, 170), (50, 160, 16, 210), (140, 160, 22, 185)]

for cx, cy, r, intensity in cells:
    add_cell(cell_image, cx, cy, r, intensity)

# Add some noise
cell_image = cell_image + np.random.randint(0, 20, (200, 200), dtype=np.uint8)
cell_image = np.clip(cell_image, 0, 255).astype(np.uint8)

# Apply thresholding to detect cells
T_cells = 130
detected_cells = threshold_binary(cell_image, T_cells)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cell_image, cmap='gray')
plt.title('Simulated Microscopy Image')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.hist(cell_image.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
plt.axvline(x=T_cells, color='r', linestyle='--', linewidth=2, label=f'T={T_cells}')
plt.xlabel('Pixel Value')
plt.ylabel('Count')
plt.title('Histogram')
plt.legend()

plt.subplot(1, 3, 3)
plt.imshow(detected_cells, cmap='gray')
plt.title(f'Detected Cells (T={T_cells})')
plt.colorbar()

plt.suptitle('Cell Detection using Thresholding', fontsize=14)
plt.tight_layout()
plt.savefig('04_cell_detection.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("LESSON 5 COMPLETED!")
print("="*50)
print("""
What we learned:
1. Thresholding converts grayscale to binary
2. Binary: pixels >= T become 255, else 0
3. Different types: binary, truncate, to-zero
4. Choosing T: use histogram to find good value
5. Applications: cell detection, segmentation, object extraction
""")
