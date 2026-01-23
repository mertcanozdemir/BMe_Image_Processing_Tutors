"""
LESSON 4: Gamma Correction and Non-Linear Transformations
=========================================================
Biomedical Image Processing Techniques

In this lesson:
- Power-law (gamma) transformation
- Log transformation
- When to use non-linear transformations
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. GAMMA (POWER-LAW) TRANSFORMATION
# =============================================================================
# Formula: g = c * f^gamma
# Usually c = 1 and f is normalized to [0, 1]
#
# gamma < 1: brightens dark regions (expands dark, compresses bright)
# gamma > 1: darkens image (compresses dark, expands bright)
# gamma = 1: no change

print("="*50)
print("GAMMA CORRECTION")
print("="*50)
print("""
Formula: g = c * f^gamma (with f normalized to [0,1])

gamma < 1: Brightens dark regions
gamma > 1: Darkens light regions
gamma = 1: No change
""")

def gamma_correction(image, gamma):
    """
    Apply gamma correction to an image.

    Parameters:
    - image: input image (uint8)
    - gamma: gamma value

    Returns:
    - corrected image (uint8)
    """
    # Normalize to [0, 1]
    img_normalized = image.astype(np.float32) / 255.0

    # Apply gamma
    img_corrected = np.power(img_normalized, gamma)

    # Scale back to [0, 255]
    return (img_corrected * 255).astype(np.uint8)

# Create a test image with gradient
gradient = np.tile(np.linspace(0, 255, 256), (100, 1)).astype(np.uint8)

# Apply different gamma values
gamma_values = [0.3, 0.5, 1.0, 1.5, 2.5]
results = [gamma_correction(gradient, g) for g in gamma_values]

# Visualize
plt.figure(figsize=(15, 8))

for i, (img, gamma) in enumerate(zip(results, gamma_values)):
    plt.subplot(len(gamma_values), 1, i+1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(f'gamma = {gamma}')
    plt.yticks([])

plt.suptitle('Gamma Correction: g = f^gamma', fontsize=14)
plt.tight_layout()
plt.savefig('01_gamma_correction.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 2. GAMMA TRANSFORMATION CURVES
# =============================================================================

print("\n" + "="*50)
print("GAMMA TRANSFORMATION CURVES")
print("="*50)

x = np.linspace(0, 1, 256)

plt.figure(figsize=(10, 8))

gamma_values = [0.2, 0.4, 0.67, 1.0, 1.5, 2.5, 5.0]
colors = plt.cm.coolwarm(np.linspace(0, 1, len(gamma_values)))

for gamma, color in zip(gamma_values, colors):
    y = np.power(x, gamma)
    label = f'gamma = {gamma}'
    if gamma < 1:
        label += ' (brighten)'
    elif gamma > 1:
        label += ' (darken)'
    plt.plot(x, y, color=color, linewidth=2, label=label)

plt.xlabel('Input Intensity (normalized)', fontsize=12)
plt.ylabel('Output Intensity (normalized)', fontsize=12)
plt.title('Gamma Transformation Curves\n$g = f^\\gamma$', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig('02_gamma_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 3. PRACTICAL EXAMPLE: ENHANCING DARK IMAGE
# =============================================================================

print("\n" + "="*50)
print("PRACTICAL: ENHANCING DARK IMAGE")
print("="*50)

# Create a "dark" image (simulating underexposed photo)
np.random.seed(42)
dark_image = np.random.randint(0, 80, (200, 200), dtype=np.uint8)

# Apply gamma correction to brighten
enhanced = gamma_correction(dark_image, 0.4)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(dark_image, cmap='gray', vmin=0, vmax=255)
plt.title('Dark Image (underexposed)')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(enhanced, cmap='gray', vmin=0, vmax=255)
plt.title('Gamma Corrected (gamma=0.4)')
plt.colorbar()

plt.suptitle('Enhancing Dark Images with Gamma Correction', fontsize=14)
plt.tight_layout()
plt.savefig('03_dark_enhancement.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 4. LOG TRANSFORMATION
# =============================================================================
# Formula: g = c * log(1 + f)
# Used to expand dark regions while compressing bright regions
# Useful for displaying Fourier spectrum

print("\n" + "="*50)
print("LOG TRANSFORMATION")
print("="*50)
print("""
Formula: g = c * log(1 + f)

- Expands dark intensity range
- Compresses bright intensity range
- Useful for: Fourier spectrum, HDR images
""")

def log_transform(image, c=1.0):
    """
    Apply log transformation to an image.

    Parameters:
    - image: input image (uint8)
    - c: scaling constant

    Returns:
    - transformed image (uint8)
    """
    img_float = image.astype(np.float32)

    # Apply log transform
    img_log = c * np.log1p(img_float)  # log1p = log(1 + x)

    # Normalize to [0, 255]
    img_log = (img_log / img_log.max()) * 255

    return img_log.astype(np.uint8)

# Create image with wide dynamic range
wide_range = np.zeros((200, 200), dtype=np.uint8)
# Create exponentially increasing values
for i in range(200):
    wide_range[:, i] = min(255, int(np.exp(i/30)))

# Apply log transform
log_result = log_transform(wide_range)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(wide_range, cmap='gray', vmin=0, vmax=255)
plt.title('Original (exponential growth)')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(log_result, cmap='gray', vmin=0, vmax=255)
plt.title('Log Transformed')
plt.colorbar()

plt.suptitle('Log Transformation: g = c * log(1 + f)', fontsize=14)
plt.tight_layout()
plt.savefig('04_log_transform.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 5. COMPARISON OF TRANSFORMATIONS
# =============================================================================

print("\n" + "="*50)
print("COMPARISON OF TRANSFORMATIONS")
print("="*50)

x = np.linspace(0, 255, 256)

plt.figure(figsize=(10, 6))

# Linear
plt.plot(x, x, 'k-', linewidth=2, label='Linear (identity)')

# Brightness
plt.plot(x, np.clip(x + 50, 0, 255), 'b--', linewidth=2, label='Brightness (+50)')

# Contrast
plt.plot(x, np.clip(x * 1.5, 0, 255), 'g--', linewidth=2, label='Contrast (1.5x)')

# Gamma < 1
gamma_result = np.power(x/255, 0.5) * 255
plt.plot(x, gamma_result, 'r-', linewidth=2, label='Gamma (0.5)')

# Gamma > 1
gamma_result2 = np.power(x/255, 2.0) * 255
plt.plot(x, gamma_result2, 'm-', linewidth=2, label='Gamma (2.0)')

# Log
log_result = np.log1p(x)
log_result = (log_result / log_result.max()) * 255
plt.plot(x, log_result, 'c-', linewidth=2, label='Log')

plt.xlabel('Input Pixel Value', fontsize=12)
plt.ylabel('Output Pixel Value', fontsize=12)
plt.title('Comparison of Different Transformations', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 255])
plt.ylim([0, 280])
plt.savefig('05_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("LESSON 4 COMPLETED!")
print("="*50)
print("""
What we learned:
1. Gamma correction: g = f^gamma
   - gamma < 1: brightens dark regions
   - gamma > 1: darkens light regions
2. Log transform: g = c * log(1 + f)
   - Expands dark, compresses bright
3. When to use:
   - Underexposed image: gamma < 1
   - Overexposed image: gamma > 1
   - High dynamic range: log transform
""")
