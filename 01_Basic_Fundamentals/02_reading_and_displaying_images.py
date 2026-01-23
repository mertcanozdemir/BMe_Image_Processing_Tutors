"""
LESSON 2: Reading and Displaying Images
========================================
Biomedical Image Processing Techniques

In this lesson:
- How to read images from files
- How to display images
- Basic image properties
- Color vs Grayscale images
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# =============================================================================
# 1. READING IMAGES WITH DIFFERENT LIBRARIES
# =============================================================================

print("="*50)
print("READING IMAGES")
print("="*50)

# Method 1: Using PIL (Pillow)
# PIL is simple and widely used
try:
    img_pil = Image.open('../ornek_goruntuler/sample.jpg')
    print(f"PIL - Size: {img_pil.size}, Mode: {img_pil.mode}")
except:
    # Create a sample image if file doesn't exist
    img_pil = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    print("Created random sample image")

# Method 2: Using OpenCV
# OpenCV is fast and has many features
# Note: OpenCV reads images in BGR format, not RGB!
try:
    img_cv2 = cv2.imread('../ornek_goruntuler/sample.jpg')
    if img_cv2 is not None:
        print(f"OpenCV - Shape: {img_cv2.shape}, dtype: {img_cv2.dtype}")
except:
    img_cv2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

# Method 3: Using matplotlib
try:
    img_plt = plt.imread('../ornek_goruntuler/sample.jpg')
    print(f"Matplotlib - Shape: {img_plt.shape}, dtype: {img_plt.dtype}")
except:
    img_plt = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

# =============================================================================
# 2. IMAGE PROPERTIES
# =============================================================================

print("\n" + "="*50)
print("IMAGE PROPERTIES")
print("="*50)

# Create a sample image for demonstration
sample_color = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
sample_gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

print("\nColor Image Properties:")
print(f"  Shape: {sample_color.shape}")
print(f"  Height: {sample_color.shape[0]} pixels")
print(f"  Width: {sample_color.shape[1]} pixels")
print(f"  Channels: {sample_color.shape[2]} (RGB)")
print(f"  Data type: {sample_color.dtype}")
print(f"  Total pixels: {sample_color.shape[0] * sample_color.shape[1]}")

print("\nGrayscale Image Properties:")
print(f"  Shape: {sample_gray.shape}")
print(f"  Height: {sample_gray.shape[0]} pixels")
print(f"  Width: {sample_gray.shape[1]} pixels")
print(f"  Channels: 1 (grayscale)")
print(f"  Data type: {sample_gray.dtype}")

# =============================================================================
# 3. COLOR vs GRAYSCALE
# =============================================================================

print("\n" + "="*50)
print("COLOR vs GRAYSCALE")
print("="*50)

# Create a simple color image
color_img = np.zeros((100, 300, 3), dtype=np.uint8)
color_img[:, 0:100, 0] = 255    # Red channel
color_img[:, 100:200, 1] = 255  # Green channel
color_img[:, 200:300, 2] = 255  # Blue channel

# Convert to grayscale using weighted average (human perception)
# Formula: Gray = 0.299*R + 0.587*G + 0.114*B
gray_img = (0.299 * color_img[:,:,0] +
            0.587 * color_img[:,:,1] +
            0.114 * color_img[:,:,2]).astype(np.uint8)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(color_img)
plt.title('Color Image (RGB)\nShape: (100, 300, 3)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray_img, cmap='gray')
plt.title('Grayscale Image\nShape: (100, 300)')
plt.axis('off')

plt.subplot(1, 3, 3)
# Show individual channels
fig_channels = np.zeros((100, 300, 3), dtype=np.uint8)
plt.imshow(color_img[:,:,0], cmap='Reds')
plt.title('Red Channel Only')
plt.axis('off')

plt.tight_layout()
plt.savefig('02_color_vs_gray.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 4. DISPLAYING IMAGES
# =============================================================================

print("\n" + "="*50)
print("DISPLAYING IMAGES")
print("="*50)

# Create sample images
img1 = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
img2 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

# Different display options
plt.figure(figsize=(15, 5))

# Grayscale with different colormaps
plt.subplot(1, 4, 1)
plt.imshow(img1, cmap='gray')
plt.title('cmap="gray"')
plt.colorbar()

plt.subplot(1, 4, 2)
plt.imshow(img1, cmap='hot')
plt.title('cmap="hot"')
plt.colorbar()

plt.subplot(1, 4, 3)
plt.imshow(img1, cmap='jet')
plt.title('cmap="jet"')
plt.colorbar()

plt.subplot(1, 4, 4)
plt.imshow(img1, cmap='viridis')
plt.title('cmap="viridis"')
plt.colorbar()

plt.suptitle('Same Image with Different Colormaps', fontsize=14)
plt.tight_layout()
plt.savefig('02_colormaps.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 5. SAVING IMAGES
# =============================================================================

print("\n" + "="*50)
print("SAVING IMAGES")
print("="*50)

# Create a test image
test_img = np.zeros((100, 100, 3), dtype=np.uint8)
test_img[25:75, 25:75] = [255, 0, 0]  # Red square

# Save with PIL
pil_img = Image.fromarray(test_img)
pil_img.save('test_pil.png')
print("Saved with PIL: test_pil.png")

# Save with OpenCV (remember BGR!)
cv2.imwrite('test_cv2.png', cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
print("Saved with OpenCV: test_cv2.png")

# Save with matplotlib
plt.imsave('test_plt.png', test_img)
print("Saved with matplotlib: test_plt.png")

print("\n" + "="*50)
print("LESSON 2 COMPLETED!")
print("="*50)
print("""
What we learned:
1. Read images with PIL, OpenCV, or matplotlib
2. Image shape: (height, width) for grayscale, (height, width, 3) for color
3. OpenCV uses BGR, others use RGB
4. Colormaps help visualize grayscale data
5. Save images with various libraries
""")
