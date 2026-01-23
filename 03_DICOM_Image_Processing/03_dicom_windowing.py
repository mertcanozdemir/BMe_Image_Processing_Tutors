"""
LESSON 3: DICOM Window/Level (Windowing)
Biomedical Image Processing - DICOM Module

Topics:
- Understanding Window Width and Window Level
- Common window presets for CT
- Interactive windowing
- Creating custom window functions
"""

import numpy as np
import matplotlib.pyplot as plt

print("Libraries imported successfully!")

# ============================================================
# 1. What is Windowing?
# ============================================================
"""
Medical images have high dynamic range (e.g., CT: -1000 to +3000 HU),
but displays show only 256 gray levels.

Windowing maps a selected range of values to the display range.

Key Concepts:
- Window Level (WL/Center): The center of the displayed range
- Window Width (WW): The range of values to display

Display Range: [WL - WW/2, WL + WW/2]
"""

# ============================================================
# 2. Create a CT Phantom for Demonstration
# ============================================================

def create_ct_phantom():
    """Create a synthetic CT-like image with different tissues."""
    np.random.seed(42)
    img = np.zeros((256, 256), dtype=np.float32)

    # Background: Air (-1000 HU)
    img[:, :] = -1000

    # Body outline: Soft tissue (~40 HU)
    y, x = np.ogrid[:256, :256]
    body = ((x - 128)**2 + (y - 128)**2) < 100**2
    img[body] = 40 + np.random.randn(np.sum(body)) * 10

    # Lung regions: Air in lungs (-500 HU)
    lung_l = ((x - 90)**2 + (y - 120)**2) < 35**2
    lung_r = ((x - 166)**2 + (y - 120)**2) < 35**2
    img[lung_l | lung_r] = -500 + np.random.randn(np.sum(lung_l | lung_r)) * 50

    # Heart: Blood (~40-50 HU)
    heart = ((x - 140)**2 / 400 + (y - 140)**2 / 600) < 1
    img[heart] = 45 + np.random.randn(np.sum(heart)) * 5

    # Spine: Bone (~400-700 HU)
    spine = ((x - 128)**2 + (y - 200)**2) < 20**2
    img[spine] = 500 + np.random.randn(np.sum(spine)) * 50

    # Ribs: Bone
    for angle in [30, 60, 120, 150]:
        rad = np.deg2rad(angle)
        cx, cy = 128 + 80 * np.cos(rad), 128 + 80 * np.sin(rad)
        rib = ((x - cx)**2 + (y - cy)**2) < 8**2
        img[rib] = 600 + np.random.randn(np.sum(rib)) * 30

    # Fat layer: (-100 HU)
    fat = (((x - 128)**2 + (y - 128)**2) < 105**2) & (((x - 128)**2 + (y - 128)**2) > 95**2)
    fat = fat & body
    img[fat] = -80 + np.random.randn(np.sum(fat)) * 10

    return img

ct_image = create_ct_phantom()
print(f"CT Image created: {ct_image.shape}")
print(f"HU range: [{ct_image.min():.0f}, {ct_image.max():.0f}]")

# ============================================================
# 3. Window/Level Function
# ============================================================

def apply_window(image, window_center, window_width):
    """
    Apply window/level to an image.

    Parameters:
    -----------
    image : ndarray
        Input image (e.g., in Hounsfield Units)
    window_center : float
        Center of the window (Level)
    window_width : float
        Width of the window

    Returns:
    --------
    ndarray : Image scaled to 0-255 range
    """
    # Calculate min and max of window
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2

    # Apply window
    windowed = np.clip(image, img_min, img_max)

    # Normalize to 0-255
    windowed = ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)

    return windowed

# Test with soft tissue window
soft_tissue = apply_window(ct_image, window_center=40, window_width=400)
print(f"\nWindowed image range: [{soft_tissue.min()}, {soft_tissue.max()}]")

# ============================================================
# 4. Common CT Window Presets
# ============================================================

CT_WINDOWS = {
    'Soft Tissue': {'center': 40, 'width': 400},
    'Lung': {'center': -500, 'width': 1500},
    'Bone': {'center': 400, 'width': 1800},
    'Brain': {'center': 40, 'width': 80},
    'Liver': {'center': 60, 'width': 150},
    'Mediastinum': {'center': 50, 'width': 350},
}

print("\nCT Window Presets:")
print("=" * 40)
for name, params in CT_WINDOWS.items():
    c, w = params['center'], params['width']
    print(f"{name:15s}: L={c:5d}, W={w:5d}  [{c-w//2:6d} to {c+w//2:5d}]")

# Compare different windows
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, (name, params) in zip(axes, CT_WINDOWS.items()):
    windowed = apply_window(ct_image, params['center'], params['width'])
    ax.imshow(windowed, cmap='gray')
    ax.set_title(f"{name}\nW:{params['width']} L:{params['center']}")
    ax.axis('off')

plt.suptitle('Same CT Image with Different Window Presets', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 5. Window Transfer Function Visualization
# ============================================================

def plot_window_transfer(window_center, window_width, ax=None):
    """Plot the transfer function for a given window setting."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # HU range
    hu = np.linspace(-1000, 1000, 2000)

    # Calculate output
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    output = np.clip(hu, img_min, img_max)
    output = (output - img_min) / (img_max - img_min) * 255

    ax.plot(hu, output, 'b-', linewidth=2)
    ax.axvline(window_center, color='r', linestyle='--', label=f'Center (L={window_center})')
    ax.axvline(img_min, color='g', linestyle=':', label=f'Min={img_min:.0f}')
    ax.axvline(img_max, color='g', linestyle=':', label=f'Max={img_max:.0f}')

    ax.set_xlabel('Input (Hounsfield Units)')
    ax.set_ylabel('Output (Display Value)')
    ax.set_title(f'Window Transfer Function\nW={window_width}, L={window_center}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-10, 265)

    return ax

# Compare transfer functions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

plot_window_transfer(40, 400, axes[0])   # Soft tissue
plot_window_transfer(-500, 1500, axes[1])  # Lung
plot_window_transfer(400, 1800, axes[2])   # Bone

axes[0].set_title('Soft Tissue Window\nW=400, L=40')
axes[1].set_title('Lung Window\nW=1500, L=-500')
axes[2].set_title('Bone Window\nW=1800, L=400')

plt.tight_layout()
plt.show()

# ============================================================
# 6. Effect of Window Width and Level
# ============================================================

# Demonstrate effect of window width
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

widths = [100, 400, 1000, 2000]
center = 40

for ax, width in zip(axes, widths):
    windowed = apply_window(ct_image, center, width)
    ax.imshow(windowed, cmap='gray')
    ax.set_title(f'Width = {width}\n(Center = {center})')
    ax.axis('off')

plt.suptitle('Effect of Window Width (Contrast)', fontsize=14)
plt.tight_layout()
plt.show()

# Demonstrate effect of window level
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

centers = [-500, -100, 40, 400]
width = 400

for ax, center in zip(axes, centers):
    windowed = apply_window(ct_image, center, width)
    ax.imshow(windowed, cmap='gray')
    ax.set_title(f'Center = {center}\n(Width = {width})')
    ax.axis('off')

plt.suptitle('Effect of Window Level (Brightness)', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 7. Auto-Windowing
# ============================================================

def auto_window(image, percentile_low=1, percentile_high=99):
    """
    Automatically determine window settings based on percentiles.

    Parameters:
    -----------
    image : ndarray
        Input image
    percentile_low : float
        Lower percentile (default 1)
    percentile_high : float
        Upper percentile (default 99)

    Returns:
    --------
    tuple : (window_center, window_width)
    """
    p_low = np.percentile(image, percentile_low)
    p_high = np.percentile(image, percentile_high)

    window_center = (p_high + p_low) / 2
    window_width = p_high - p_low

    return window_center, window_width

# Test auto-windowing
auto_center, auto_width = auto_window(ct_image)
print(f"\nAuto-detected window: L={auto_center:.0f}, W={auto_width:.0f}")

# Display
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Auto window
windowed_auto = apply_window(ct_image, auto_center, auto_width)
axes[0].imshow(windowed_auto, cmap='gray')
axes[0].set_title(f'Auto Window\nL={auto_center:.0f}, W={auto_width:.0f}')
axes[0].axis('off')

# Soft tissue preset
windowed_preset = apply_window(ct_image, 40, 400)
axes[1].imshow(windowed_preset, cmap='gray')
axes[1].set_title('Soft Tissue Preset\nL=40, W=400')
axes[1].axis('off')

plt.suptitle('Auto vs Preset Windowing', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 8. Sigmoid Windowing
# ============================================================

def apply_sigmoid_window(image, window_center, window_width):
    """
    Apply sigmoid (S-curve) windowing.

    Provides smoother transitions at window boundaries.
    """
    # Normalize to -4 to 4 range for sigmoid
    x = (image - window_center) / (window_width / 8)

    # Apply sigmoid
    sigmoid = 1 / (1 + np.exp(-x))

    # Scale to 0-255
    return (sigmoid * 255).astype(np.uint8)

# Compare linear vs sigmoid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Transfer functions
hu = np.linspace(-500, 500, 1000)
wc, ww = 40, 400

# Linear transfer
linear_out = np.clip((hu - (wc - ww/2)) / ww * 255, 0, 255)

# Sigmoid transfer
x = (hu - wc) / (ww / 8)
sigmoid_out = 1 / (1 + np.exp(-x)) * 255

axes[0, 0].plot(hu, linear_out, 'b-', label='Linear', linewidth=2)
axes[0, 0].plot(hu, sigmoid_out, 'r-', label='Sigmoid', linewidth=2)
axes[0, 0].set_xlabel('Hounsfield Units')
axes[0, 0].set_ylabel('Output Value')
axes[0, 0].set_title('Transfer Functions')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Empty plot for layout
axes[0, 1].axis('off')

# Linear windowed image
linear_img = apply_window(ct_image, wc, ww)
axes[1, 0].imshow(linear_img, cmap='gray')
axes[1, 0].set_title('Linear Window')
axes[1, 0].axis('off')

# Sigmoid windowed image
sigmoid_img = apply_sigmoid_window(ct_image, wc, ww)
axes[1, 1].imshow(sigmoid_img, cmap='gray')
axes[1, 1].set_title('Sigmoid Window')
axes[1, 1].axis('off')

plt.suptitle('Linear vs Sigmoid Windowing', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# 9. CT Window Manager Class
# ============================================================

class CTWindowManager:
    """
    Utility class for CT image windowing.
    """

    PRESETS = {
        'brain': {'center': 40, 'width': 80},
        'stroke': {'center': 40, 'width': 40},
        'lung': {'center': -500, 'width': 1500},
        'mediastinum': {'center': 50, 'width': 350},
        'soft_tissue': {'center': 40, 'width': 400},
        'liver': {'center': 60, 'width': 150},
        'bone': {'center': 400, 'width': 1800},
    }

    @staticmethod
    def apply(image, center, width):
        """Apply window to image."""
        img_min = center - width / 2
        img_max = center + width / 2
        windowed = np.clip(image, img_min, img_max)
        return ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)

    @classmethod
    def apply_preset(cls, image, preset_name):
        """Apply a preset window."""
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(cls.PRESETS.keys())}")
        params = cls.PRESETS[preset_name]
        return cls.apply(image, params['center'], params['width'])

    @staticmethod
    def auto_window(image, percentile=(1, 99)):
        """Auto-detect window parameters."""
        p_low = np.percentile(image, percentile[0])
        p_high = np.percentile(image, percentile[1])
        center = (p_high + p_low) / 2
        width = p_high - p_low
        return center, width

# Test the class
print("\nAvailable presets:", list(CTWindowManager.PRESETS.keys()))

# Apply lung preset
lung_img = CTWindowManager.apply_preset(ct_image, 'lung')
print(f"Lung preset applied: {lung_img.shape}, range [{lung_img.min()}, {lung_img.max()}]")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
What we learned:
1. Window/Level maps high dynamic range to displayable values
2. Window Width controls contrast (narrow = high contrast)
3. Window Level controls brightness (center of display range)
4. Presets exist for different anatomical regions
5. Linear windowing is most common, sigmoid provides smoother transitions
6. Auto-windowing can help find reasonable settings automatically
""")
