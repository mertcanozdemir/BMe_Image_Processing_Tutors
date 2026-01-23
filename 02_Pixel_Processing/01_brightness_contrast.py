"""
LESSON 3: Brightness and Contrast Adjustment
=============================================
Biomedical Image Processing Techniques

In this lesson:
- What is brightness
- What is contrast
- How to adjust them mathematically
- Point operations on images
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. POINT OPERATIONS (Pixel-wise Operations)
# =============================================================================
# Point operations transform each pixel independently
# Output pixel depends ONLY on the corresponding input pixel
# Formula: g(x,y) = T[f(x,y)]

print("="*50)
print("POINT OPERATIONS")
print("="*50)
print("""
Point Operation: g(x,y) = T[f(x,y)]

Where:
- f(x,y) = input pixel value at position (x,y)
- g(x,y) = output pixel value at position (x,y)
- T = transformation function
""")

# Create a sample grayscale image
np.random.seed(42)
original = np.random.randint(50, 200, (200, 200), dtype=np.uint8)

# =============================================================================
# 2. BRIGHTNESS ADJUSTMENT
# =============================================================================
# Brightness = adding/subtracting a constant value
# Formula: g(x,y) = f(x,y) + b
# b > 0: brighter image
# b < 0: darker image

print("\n" + "="*50)
print("BRIGHTNESS ADJUSTMENT")
print("="*50)
print("Formula: g(x,y) = f(x,y) + b")

def adjust_brightness(image, value):
    """
    Adjust brightness by adding a constant value.

    Parameters:
    - image: input image (uint8)
    - value: brightness adjustment (-255 to 255)

    Returns:
    - adjusted image (uint8)
    """
    # Convert to float to avoid overflow
    img_float = image.astype(np.float32)

    # Add brightness value
    img_float = img_float + value

    # Clip values to valid range [0, 255]
    img_float = np.clip(img_float, 0, 255)

    return img_float.astype(np.uint8)

# Apply different brightness values
bright_plus50 = adjust_brightness(original, 50)
bright_minus50 = adjust_brightness(original, -50)

# Visualize
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(bright_minus50, cmap='gray', vmin=0, vmax=255)
plt.title('Darker (b = -50)')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(original, cmap='gray', vmin=0, vmax=255)
plt.title('Original')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(bright_plus50, cmap='gray', vmin=0, vmax=255)
plt.title('Brighter (b = +50)')
plt.colorbar()

plt.suptitle('Brightness Adjustment: g(x,y) = f(x,y) + b', fontsize=14)
plt.tight_layout()
plt.savefig('01_brightness.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 3. CONTRAST ADJUSTMENT
# =============================================================================
# Contrast = multiplying by a constant value
# Formula: g(x,y) = c * f(x,y)
# c > 1: higher contrast
# c < 1: lower contrast
# c = 1: no change

print("\n" + "="*50)
print("CONTRAST ADJUSTMENT")
print("="*50)
print("Formula: g(x,y) = c * f(x,y)")

def adjust_contrast(image, factor):
    """
    Adjust contrast by multiplying with a constant factor.

    Parameters:
    - image: input image (uint8)
    - factor: contrast factor (0.0 to 3.0 typical)

    Returns:
    - adjusted image (uint8)
    """
    img_float = image.astype(np.float32)
    img_float = img_float * factor
    img_float = np.clip(img_float, 0, 255)
    return img_float.astype(np.uint8)

# Apply different contrast values
contrast_low = adjust_contrast(original, 0.5)
contrast_high = adjust_contrast(original, 1.5)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(contrast_low, cmap='gray', vmin=0, vmax=255)
plt.title('Low Contrast (c = 0.5)')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(original, cmap='gray', vmin=0, vmax=255)
plt.title('Original (c = 1.0)')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(contrast_high, cmap='gray', vmin=0, vmax=255)
plt.title('High Contrast (c = 1.5)')
plt.colorbar()

plt.suptitle('Contrast Adjustment: g(x,y) = c * f(x,y)', fontsize=14)
plt.tight_layout()
plt.savefig('02_contrast.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 4. COMBINED BRIGHTNESS AND CONTRAST
# =============================================================================
# Formula: g(x,y) = c * f(x,y) + b
# This is a LINEAR transformation

print("\n" + "="*50)
print("COMBINED: BRIGHTNESS + CONTRAST")
print("="*50)
print("Formula: g(x,y) = c * f(x,y) + b")

def adjust_brightness_contrast(image, contrast=1.0, brightness=0):
    """
    Adjust both brightness and contrast.

    Parameters:
    - image: input image (uint8)
    - contrast: contrast factor
    - brightness: brightness offset

    Returns:
    - adjusted image (uint8)
    """
    img_float = image.astype(np.float32)
    img_float = contrast * img_float + brightness
    img_float = np.clip(img_float, 0, 255)
    return img_float.astype(np.uint8)

# Different combinations
result1 = adjust_brightness_contrast(original, contrast=1.5, brightness=30)
result2 = adjust_brightness_contrast(original, contrast=0.7, brightness=-20)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original, cmap='gray', vmin=0, vmax=255)
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(result1, cmap='gray', vmin=0, vmax=255)
plt.title('c=1.5, b=+30\n(High contrast, brighter)')

plt.subplot(1, 3, 3)
plt.imshow(result2, cmap='gray', vmin=0, vmax=255)
plt.title('c=0.7, b=-20\n(Low contrast, darker)')

plt.suptitle('Combined: g(x,y) = c * f(x,y) + b', fontsize=14)
plt.tight_layout()
plt.savefig('03_combined.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 5. VISUALIZATION OF TRANSFORMATION FUNCTION
# =============================================================================

print("\n" + "="*50)
print("TRANSFORMATION FUNCTIONS")
print("="*50)

x = np.arange(0, 256)

plt.figure(figsize=(12, 5))

# Brightness transformations
plt.subplot(1, 2, 1)
plt.plot(x, np.clip(x + 50, 0, 255), 'b-', label='b = +50 (brighter)')
plt.plot(x, x, 'k--', label='Original (b = 0)')
plt.plot(x, np.clip(x - 50, 0, 255), 'r-', label='b = -50 (darker)')
plt.xlabel('Input Pixel Value')
plt.ylabel('Output Pixel Value')
plt.title('Brightness Transformation\ng(x,y) = f(x,y) + b')
plt.legend()
plt.grid(True)
plt.xlim([0, 255])
plt.ylim([0, 255])

# Contrast transformations
plt.subplot(1, 2, 2)
plt.plot(x, np.clip(x * 1.5, 0, 255), 'b-', label='c = 1.5 (high)')
plt.plot(x, x, 'k--', label='Original (c = 1.0)')
plt.plot(x, np.clip(x * 0.5, 0, 255), 'r-', label='c = 0.5 (low)')
plt.xlabel('Input Pixel Value')
plt.ylabel('Output Pixel Value')
plt.title('Contrast Transformation\ng(x,y) = c * f(x,y)')
plt.legend()
plt.grid(True)
plt.xlim([0, 255])
plt.ylim([0, 255])

plt.tight_layout()
plt.savefig('04_transformation_functions.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("LESSON 3 COMPLETED!")
print("="*50)
print("""
What we learned:
1. Point operations transform pixels independently
2. Brightness: g = f + b (shift values up/down)
3. Contrast: g = c * f (stretch/compress values)
4. Combined: g = c * f + b (linear transformation)
5. Clipping prevents overflow (keep values in [0,255])
""")
