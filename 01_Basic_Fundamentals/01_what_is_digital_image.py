"""
LESSON 1: What is a Digital Image?
==================================
Biomedical Image Processing Techniques

In this lesson:
- What is a digital image
- What is a pixel
- Mathematical representation of images
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. IMAGE = 2D NUMERICAL MATRIX
# =============================================================================
# A digital image is essentially a matrix of numbers.
# Each number represents a "pixel" (picture element) value.

# Simplest example: Create a small 5x5 pixel image
simple_image = np.array([
    [0,   0,   0,   0,   0],      # Black row (0 = black)
    [0, 255, 255, 255,   0],      # White in the middle (255 = white)
    [0, 255, 255, 255,   0],
    [0, 255, 255, 255,   0],
    [0,   0,   0,   0,   0]       # Black row
], dtype=np.uint8)

print("An image is actually a matrix of numbers:")
print(simple_image)
print(f"\nShape: {simple_image.shape}")  # (5, 5) - 5 rows, 5 columns

# Visualize
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(simple_image, cmap='gray', vmin=0, vmax=255)
plt.title('5x5 Pixel Image')
plt.colorbar(label='Pixel Value')

# Show pixel values on the image
plt.subplot(1, 2, 2)
plt.imshow(simple_image, cmap='gray', vmin=0, vmax=255)
for i in range(5):
    for j in range(5):
        plt.text(j, i, str(simple_image[i, j]),
                ha='center', va='center', color='red', fontsize=12)
plt.title('Pixel Values Visible')
plt.colorbar(label='Pixel Value')

plt.tight_layout()
plt.savefig('01_what_is_image.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 2. GRAYSCALE IMAGE
# =============================================================================
# In an 8-bit grayscale image:
# - Minimum value: 0 (pure black)
# - Maximum value: 255 (pure white)
# - Total of 256 different gray levels

print("\n" + "="*50)
print("GRAYSCALE SCALE")
print("="*50)

# Create a gradient from 0 to 255
gradient = np.linspace(0, 255, 256).reshape(1, 256).astype(np.uint8)
gradient = np.repeat(gradient, 50, axis=0)  # Increase height

plt.figure(figsize=(12, 3))
plt.imshow(gradient, cmap='gray', vmin=0, vmax=255)
plt.title('Grayscale: 0 (Black) → 255 (White)')
plt.xlabel('Pixel Value')
plt.yticks([])
plt.xticks([0, 64, 128, 192, 255])
plt.savefig('02_grayscale.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 3. PIXEL COORDINATES
# =============================================================================
# Pixel positions in an image are expressed as (row, column)
# Top-left corner: (0, 0)

print("\n" + "="*50)
print("PIXEL COORDINATES")
print("="*50)

# 8x8 example image
coord_example = np.zeros((8, 8), dtype=np.uint8)

# Assign values to specific pixels
coord_example[0, 0] = 255   # Top-left corner
coord_example[0, 7] = 200   # Top-right corner
coord_example[7, 0] = 150   # Bottom-left corner
coord_example[7, 7] = 100   # Bottom-right corner
coord_example[3, 4] = 180   # Somewhere in the middle

print("Example image matrix:")
print(coord_example)

print("\nAccessing pixel values:")
print(f"Top-left [0,0]: {coord_example[0, 0]}")
print(f"Bottom-right [7,7]: {coord_example[7, 7]}")
print(f"Middle pixel [3,4]: {coord_example[3, 4]}")

plt.figure(figsize=(6, 6))
plt.imshow(coord_example, cmap='gray', vmin=0, vmax=255)
plt.title('Pixel Coordinates\n(row, column)')

# Show coordinates
plt.text(0, 0, '(0,0)\n255', ha='center', va='center', color='red', fontsize=9)
plt.text(7, 0, '(0,7)\n200', ha='center', va='center', color='red', fontsize=9)
plt.text(0, 7, '(7,0)\n150', ha='center', va='center', color='yellow', fontsize=9)
plt.text(7, 7, '(7,7)\n100', ha='center', va='center', color='yellow', fontsize=9)
plt.text(4, 3, '(3,4)\n180', ha='center', va='center', color='cyan', fontsize=9)

plt.xlabel('Column (x)')
plt.ylabel('Row (y)')
plt.savefig('03_coordinates.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 4. DATA TYPES
# =============================================================================
print("\n" + "="*50)
print("IMAGE DATA TYPES")
print("="*50)

print("""
Common data types:
------------------
uint8  : 0-255 range (8-bit, most common)
uint16 : 0-65535 range (16-bit, medical images)
float32: 0.0-1.0 range (processing results)

In biomedical imaging:
- X-ray, CT: usually 12-16 bit
- MRI: usually 12-16 bit
- Ultrasound: 8 bit
- Microscopy: 8-16 bit
""")

# uint8 example
image_uint8 = np.array([[0, 128, 255]], dtype=np.uint8)
print(f"uint8 example: {image_uint8}, dtype: {image_uint8.dtype}")

# float example (normalized 0-1)
image_float = image_uint8.astype(np.float32) / 255.0
print(f"float32 (normalized): {image_float}, dtype: {image_float.dtype}")

print("\n" + "="*50)
print("LESSON 1 COMPLETED!")
print("="*50)
print("""
What we learned:
1. Digital image = 2D number matrix
2. Each number is a pixel value
3. In 8-bit image: 0=black, 255=white
4. Pixel position: expressed as (row, column)
5. Different data types for different purposes
""")
