# sharpen_bubbles.py
# Requirements: numpy, opencv-python, scikit-image, matplotlib
# pip install numpy opencv-python scikit-image matplotlib

from __future__ import annotations
import os
from typing import Tuple
import numpy as np
import cv2
from matplotlib import pyplot as plt

from skimage import img_as_float32, img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import unsharp_mask
from skimage.exposure import equalize_adapthist
from skimage.restoration import wiener, richardson_lucy


# ---------------------------
# Utilities
# ---------------------------

def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _read_image_keep_dtype(path: str) -> Tuple[np.ndarray, np.dtype]:
    """Read RGB image (BGR from cv2 then convert to RGB). Returns image and original dtype."""
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    orig_dtype = bgr.dtype
    # Convert to 3-channel RGB for processing; if grayscale, promote to 3-ch for display consistency
    if len(bgr.shape) == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, orig_dtype

def _to_float01(img: np.ndarray) -> np.ndarray:
    """Convert uint8/uint16/float to float32 in [0,1]."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    f = img.astype(np.float32)
    # best-effort clamp
    f_min, f_max = np.min(f), np.max(f)
    if f_min < 0.0 or f_max > 1.0:
        f = np.clip(f, 0.0, 1.0)
    return f

def _from_float01_like(img_float: np.ndarray, like_dtype: np.dtype) -> np.ndarray:
    """Convert back to the original dtype (preserving 'pixel format')."""
    img_float = np.clip(img_float, 0.0, 1.0)
    if like_dtype == np.uint8:
        return (img_float * 255.0 + 0.5).astype(np.uint8)
    if like_dtype == np.uint16:
        return (img_float * 65535.0 + 0.5).astype(np.uint16)
    return img_float.astype(like_dtype)

def _save_rgb(path: str, rgb: np.ndarray) -> None:
    """Save RGB using cv2 (expects BGR)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

def _variance_of_laplacian(gray_f: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Focus measure: higher = sharper."""
    lap = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=ksize)
    # Local variance via box filter (fast)
    mu = cv2.blur(lap, (5, 5))
    mu2 = cv2.blur(lap * lap, (5, 5))
    var = np.maximum(mu2 - mu * mu, 0.0)
    return var


# ---------------------------
# Methods
# ---------------------------

def method_unsharp_single_scale(rgb: np.ndarray, amount: float = 1.0, radius: float = 1.5, threshold: float = 0.0) -> np.ndarray:
    """Classic Unsharp Mask (single scale)."""
    f = _to_float01(rgb)
    # apply per-channel USM
    out = np.empty_like(f)
    for c in range(3):
        out[..., c] = unsharp_mask(f[..., c], radius=radius, amount=amount, preserve_range=True)
    return np.clip(out, 0, 1)

def method_unsharp_multi_scale(rgb: np.ndarray, radii=(1, 2, 4, 8), amounts=(0.8, 0.6, 0.4, 0.2)) -> np.ndarray:
    """Multi-scale USM: stack several gentle passes to reduce halos."""
    f = _to_float01(rgb)
    out = f.copy()
    for r, a in zip(radii, amounts):
        for c in range(3):
            out[..., c] = unsharp_mask(out[..., c], radius=float(r), amount=float(a), preserve_range=True)
    return np.clip(out, 0, 1)

def method_clahe_then_usm(rgb: np.ndarray, clip_limit: float = 0.01, usm_radius: float = 1.5, usm_amount: float = 0.8) -> np.ndarray:
    """CLAHE (on luminance) → mild USM."""
    f = _to_float01(rgb)
    # convert to YUV and apply CLAHE to Y
    yuv = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
    y = yuv[..., 0] / 255.0
    y_eq = equalize_adapthist(y, clip_limit=clip_limit)
    yuv[..., 0] = img_as_ubyte(y_eq)
    f_eq = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB).astype(np.float32) / 255.0
    # mild USM
    out = np.empty_like(f_eq)
    for c in range(3):
        out[..., c] = unsharp_mask(f_eq[..., c], radius=usm_radius, amount=usm_amount, preserve_range=True)
    return np.clip(out, 0, 1)

def method_wiener_gaussian(rgb: np.ndarray, sigma: float = 1.2, K: float = 0.004) -> np.ndarray:
    """Wiener deconvolution assuming small Gaussian blur (applied on luminance)."""
    f = _to_float01(rgb)
    gray = rgb2gray(f)  # float in [0,1]
    # build Gaussian PSF
    size = int(2 * round(3 * sigma) + 1)
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    psf = np.exp(-(X**2 + Y**2) / (2 * sigma * sigma))
    psf /= psf.sum() + 1e-8
    deconv = np.clip(wiener(gray, psf, balance=K, clip=False), 0, 1)
    # replace luminance while preserving color via ratio
    eps = 1e-6
    ratio = (deconv + eps) / (gray + eps)
    out = np.clip(f * ratio[..., None], 0, 1)
    return out

def method_rl_disk(rgb: np.ndarray, radius_px: int = 3, iterations: int = 20) -> np.ndarray:
    """Richardson–Lucy with a disk PSF (defocus-like blur) on luminance."""
    f = _to_float01(rgb)
    gray = rgb2gray(f)
    # simple disk PSF
    size = radius_px * 2 + 1
    Y, X = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
    mask = X**2 + Y**2 <= radius_px**2
    psf = np.zeros((size, size), dtype=np.float32)
    psf[mask] = 1.0
    psf /= psf.sum() + 1e-8
    deconv = np.clip(richardson_lucy(gray, psf, num_iter=iterations, clip=False), 0, 1)
    # luminance replace with color preservation
    eps = 1e-6
    ratio = (deconv + eps) / (gray + eps)
    out = np.clip(f * ratio[..., None], 0, 1)
    return out

def method_selective_defocus_rl(rgb: np.ndarray,
                                defocus_threshold: float = 0.0008,
                                disk_radius_if_blur: int = 3,
                                iterations: int = 15) -> np.ndarray:
    """
    Estimate defocus map via variance of Laplacian.
    Apply RL deconvolution only where defocus < threshold (blurred areas).
    Edge-aware (bilateral) blend back to avoid halos on already-sharp regions.
    """
    f = _to_float01(rgb)
    gray = rgb2gray(f).astype(np.float32)
    # defocus map: lower var-Lap => blurrier
    varlap = _variance_of_laplacian(gray, ksize=3)
    # normalize to [0,1]
    varlap_n = (varlap - varlap.min()) / (varlap.max() - varlap.min() + 1e-8)
    blur_mask = (varlap_n < defocus_threshold).astype(np.float32)

    # smooth mask to avoid seams
    blur_mask = cv2.GaussianBlur(blur_mask, (0, 0), 1.2)
    blur_mask = np.clip(blur_mask, 0.0, 1.0)

    # RL deconv on whole image luminance (simpler/fast), then blend by mask
    rl = method_rl_disk(rgb, radius_px=disk_radius_if_blur, iterations=iterations)
    # edge-aware feathering via bilateral filter on the mask
    # (use bilateral on mask to align to edges)
    m = cv2.bilateralFilter((blur_mask * 255).astype(np.uint8), d=9, sigmaColor=50, sigmaSpace=7).astype(np.float32) / 255.0
    m = m[..., None]  # to 3-ch
    out = rl * m + f * (1.0 - m)
    return np.clip(out, 0, 1)


# ---------------------------
# Pipeline Runner
# ---------------------------

def run_all_methods(img_rgb_path: str, output_dir: str) -> dict:
    _ensure_dir(output_dir)
    rgb, orig_dtype = _read_image_keep_dtype(img_rgb_path)

    results = {}
    results["original"] = rgb

    # 1) Unsharp (single scale)
    usm1 = method_unsharp_single_scale(rgb, amount=1.0, radius=1.5, threshold=0.0)
    results["usm_single"] = _from_float01_like(usm1, orig_dtype)

    # 2) Unsharp (multi-scale)
    usm_ms = method_unsharp_multi_scale(rgb, radii=(1, 2, 4, 8), amounts=(0.8, 0.6, 0.4, 0.2))
    results["usm_multiscale"] = _from_float01_like(usm_ms, orig_dtype)

    # 3) CLAHE → USM
    clahe_usm = method_clahe_then_usm(rgb, clip_limit=0.01, usm_radius=1.5, usm_amount=0.8)
    results["clahe_usm"] = _from_float01_like(clahe_usm, orig_dtype)

    # 4) Wiener (Gaussian)
    wien = method_wiener_gaussian(rgb, sigma=1.2, K=0.004)
    results["wiener_gaussian"] = _from_float01_like(wien, orig_dtype)

    # 5) Richardson–Lucy with disk PSF
    rl = method_rl_disk(rgb, radius_px=3, iterations=20)
    results["rl_disk"] = _from_float01_like(rl, orig_dtype)

    # 6) Selective RL guided by defocus
    selective = method_selective_defocus_rl(rgb,
                                            defocus_threshold=0.25,  # 0..1 after normalization; tweak based on image
                                            disk_radius_if_blur=3,
                                            iterations=15)
    results["selective_rl"] = _from_float01_like(selective, orig_dtype)

    # Save results
    for name, img in results.items():
        _save_rgb(os.path.join(output_dir, f"{name}.png"), img)

    return results

def plot_comparison(results: dict, cols: int = 3, figsize: Tuple[int, int] = (18, 12)) -> None:
    """Plot all results on one canvas."""
    names = list(results.keys())
    imgs = [results[k] for k in names]
    rows = int(np.ceil(len(imgs) / cols))
    plt.figure(figsize=figsize)
    for i, (nm, im) in enumerate(zip(names, imgs), 1):
        plt.subplot(rows, cols, i)
        plt.imshow(im)
        plt.title(nm, fontsize=12)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    import os
    from matplotlib import pyplot as plt

    # Define paths
    img_rgb_path = "../../tests/test_image_raw_30ppm.JPG"
    output_dir = "../../tests/sharpened_results"

    results = run_all_methods(img_rgb_path, output_dir)
    plot_comparison(results, cols=3, figsize=(18, 12))
    print(f"Saved results to: {os.path.abspath(output_dir)}")
