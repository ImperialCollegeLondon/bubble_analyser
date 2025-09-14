import os
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass

# Use built-in libraries
from scipy import stats
from scipy.signal import savgol_filter, find_peaks
from scipy.signal.windows import tukey
from scipy.stats import theilslopes
from skimage.transform import resize

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

@dataclass
class PSDParams:
    patch_size: int = 128
    overlap: float = 0.5
    tukey_alpha: float = 0.25
    radial_bins: int = 32
    slope_energy_bounds: Tuple[float, float] = (0.15, 0.85)
    resize_to: Optional[int] = 512
    eps: float = 1e-12


class PSDAnalyzer:
    """2D Power Spectral Density analyzer with Welch periodogram and radial averaging."""
    
    def __init__(self, params: PSDParams = None): # type: ignore
        self.params = params if params is not None else PSDParams()
        self.per_image_features = []
        
    def _to_grayscale_float01(self, img: np.ndarray) -> np.ndarray:
        """Ensure grayscale float32 in [0,1]. Accepts HxW or HxWxC arrays."""
        if img.ndim == 3:
            img = 0.2989 * img[...,0] + 0.5870 * img[...,1] + 0.1140 * img[...,2]
        img = img.astype(np.float32)
        minv, maxv = np.min(img), np.max(img)
        if maxv > minv:
            img = (img - minv) / (maxv - minv)
        else:
            img = np.zeros_like(img, dtype=np.float32)
        return img

    def _resize_center_crop(self, img: np.ndarray, target: int) -> np.ndarray:
        """Resize the smaller side to target, then center-crop to a square."""
        H, W = img.shape
        if min(H, W) == target and H == W:
            return img.copy()
        
        if H < W:
            newH = target
            newW = int(round(W * (target / H)))
        else:
            newW = target
            newH = int(round(H * (target / W)))
        
        # Simple nearest neighbor interpolation
        y_idx = np.linspace(0, H - 1, newH).astype(np.int32)
        x_idx = np.linspace(0, W - 1, newW).astype(np.int32)
        arr = img[np.ix_(y_idx, x_idx)]
        
        # Center crop
        H2, W2 = arr.shape
        top = max(0, (H2 - target) // 2)
        left = max(0, (W2 - target) // 2)
        return arr[top:top+target, left:left+target]

    def _tukey_window(self, n: int, alpha: float) -> np.ndarray:
        """1D Tukey window using NumPy."""
        if alpha <= 0:
            return np.ones(n, dtype=np.float32)
        if alpha >= 1:
            return np.hanning(n).astype(np.float32)
        
        w = np.ones(n, dtype=np.float32)
        edge = int(alpha * (n - 1) / 2.0)
        n1 = edge
        n2 = n - edge
        
        for i in range(n):
            if i < n1:
                w[i] = 0.5 * (1 + math.cos(math.pi * (2 * i / (alpha * (n - 1)) - 1)))
            elif i > n2:
                w[i] = 0.5 * (1 + math.cos(math.pi * (2 * (n - 1 - i) / (alpha * (n - 1)) - 1)))
        return w

    def _tukey_window_2d(self, h: int, w: int, alpha: float) -> np.ndarray:
        """2D Tukey window."""
        wy = self._tukey_window(h, alpha)
        wx = self._tukey_window(w, alpha)
        return np.outer(wy, wx).astype(np.float32)

    def _robust_slope(self, logf: np.ndarray, logP: np.ndarray) -> float:
        """Robust slope using median of pairwise slopes (Theil-Sen estimator)."""
        n = len(logf)
        if n < 3:
            return 0.0
        
        slopes = []
        for i in range(n):
            for j in range(i + 1, n):
                df = logf[j] - logf[i]
                if abs(df) > 1e-12:
                    slopes.append((logP[j] - logP[i]) / df)
        
        return float(np.median(slopes)) if slopes else 0.0

    def _kurtosis(self, x: np.ndarray, eps: float = None) -> float: # type: ignore
        """Calculate kurtosis using NumPy."""
        if eps is None:
            eps = self.params.eps
        x = np.asarray(x, dtype=np.float64)
        m = np.mean(x)
        s2 = np.var(x) + eps
        m4 = np.mean((x - m)**4)
        if s2 <= eps:
            return 0.0
        return float(m4 / (s2**2) - 3)  # Excess kurtosis

    def _skewness(self, x: np.ndarray, eps: float = None) -> float: # type: ignore
        """Calculate skewness using NumPy."""
        if eps is None:
            eps = self.params.eps
        x = np.asarray(x, dtype=np.float64)
        m = np.mean(x)
        s = np.sqrt(np.var(x) + eps)
        m3 = np.mean((x - m)**3)
        if s <= eps:
            return 0.0
        return float(m3 / (s**3))

    def _smooth_curve(self, y: np.ndarray, window: int = 7, poly: int = 3) -> np.ndarray:
        """Simple moving average smoothing."""
        if len(y) < window:
            return y.copy()
        
        # Simple moving average
        half = window // 2
        smoothed = np.zeros_like(y)
        for i in range(len(y)):
            start = max(0, i - half)
            end = min(len(y), i + half + 1)
            smoothed[i] = np.mean(y[start:end])
        return smoothed

    def _find_peaks_simple(self, y: np.ndarray) -> Tuple[int, float]:
        """Simple peak detection using local maxima."""
        if len(y) < 3:
            peak_idx = int(np.argmax(y))
            return peak_idx, 0.0
        
        # Find local maxima
        peaks = []
        prominences = []
        
        for i in range(1, len(y) - 1):
            if y[i] > y[i-1] and y[i] > y[i+1]:
                # Calculate prominence as difference from local minima
                left_min = np.min(y[max(0, i-5):i])
                right_min = np.min(y[i+1:min(len(y), i+6)])
                prominence = y[i] - max(left_min, right_min)
                peaks.append(i)
                prominences.append(prominence)
        
        if peaks:
            # Return peak with highest prominence
            best_idx = np.argmax(prominences)
            return peaks[best_idx], prominences[best_idx]
        else:
            # Fallback to global maximum
            peak_idx = int(np.argmax(y))
            return peak_idx, 0.0

    def welch_periodogram_2d(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 2D Welch-averaged periodogram."""
        img = self._to_grayscale_float01(img)
        img = img - np.mean(img)
        
        if self.params.resize_to is not None:
            img = self._resize_center_crop(img, self.params.resize_to)
        
        H, W = img.shape
        ps = self.params.patch_size
        step = max(1, int(ps * (1.0 - self.params.overlap)))
        wyx = self._tukey_window_2d(ps, ps, self.params.tukey_alpha)

        fy = np.fft.fftfreq(ps, d=1.0)
        fx = np.fft.fftfreq(ps, d=1.0)
        fy = np.fft.fftshift(fy)
        fx = np.fft.fftshift(fx)

        P2 = np.zeros((ps, ps), dtype=np.float64)
        count = 0
        
        for y in range(0, max(1, H - ps + 1), step):
            for x in range(0, max(1, W - ps + 1), step):
                patch = img[y:y+ps, x:x+ps]
                if patch.shape != (ps, ps):
                    continue
                patch = patch * wyx
                F = np.fft.fft2(patch)
                S = np.abs(F)**2
                P2 += S
                count += 1
        
        if count == 0:
            patch = img[:ps, :ps]
            patch = patch * wyx
            F = np.fft.fft2(patch)
            S = np.abs(F)**2
            P2 = S
            count = 1
        
        P2 /= count
        P2 = np.fft.fftshift(P2)
        return P2.astype(np.float64), fy, fx

    def radial_psd(self, P2: np.ndarray, fy: np.ndarray, fx: np.ndarray, K: int = None) -> Tuple[np.ndarray, np.ndarray]: # type: ignore
        """Radially average the 2D power spectrum."""
        if K is None:
            K = self.params.radial_bins
        eps = self.params.eps
        
        H, W = P2.shape
        yy, xx = np.meshgrid(fy, fx, indexing='ij')
        rr = np.sqrt(xx**2 + yy**2)

        r_min = max(1e-6, float(np.percentile(rr, 5)))
        r_max = np.max(rr)
        edges = np.logspace(np.log10(r_min), np.log10(r_max), K+1)
        P_radial = np.zeros(K, dtype=np.float64)
        counts = np.zeros(K, dtype=np.int64)

        flat_rr = rr.ravel()
        flat_P = P2.ravel()
        inds = np.digitize(flat_rr, edges) - 1
        
        for k in range(K):
            mask = inds == k
            if np.any(mask):
                P_radial[k] = np.mean(flat_P[mask])
                counts[k] = np.sum(mask)

        for k in range(K):
            if counts[k] == 0:
                P_radial[k] = eps

        total = np.sum(P_radial) + eps
        P_radial = P_radial / total
        f_centers = np.sqrt(edges[:-1] * edges[1:])
        return f_centers.astype(np.float64), P_radial.astype(np.float64)

    def compute_psd_features(
        self,
        img: np.ndarray,
        compute_anisotropy: bool = False,
        plot_path: Optional[str] = None,
        image_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute PSD-based features for a single image."""
        P2, fy, fx = self.welch_periodogram_2d(img)
        f, Pr = self.radial_psd(P2, fy, fx)

        # Compute slope
        cumsum = np.cumsum(Pr)
        lo, hi = self.params.slope_energy_bounds
        lo = np.clip(lo, 0.0, 0.95)
        hi = np.clip(hi, lo + 0.01, 1.0)
        f_lo = f[np.searchsorted(cumsum, lo, side='left')]
        f_hi = f[np.searchsorted(cumsum, hi, side='left')]
        mask = (f >= f_lo) & (f <= f_hi)
        logf = np.log(np.maximum(f[mask], self.params.eps))
        logP = np.log(np.maximum(Pr[mask], self.params.eps))
        slope = self._robust_slope(logf, logP)

        # Statistical moments
        kur = self._kurtosis(Pr)
        skw = self._skewness(Pr)

        # HF/LF ratio
        f_mid = np.median(f)
        hf = np.sum(Pr[f > f_mid])
        lf = np.sum(Pr[f <= f_mid]) + self.params.eps
        hf_lf = float(hf / lf)

        # Peak detection
        y = self._smooth_curve(Pr, window=7, poly=3)
        peak_idx, peak_prominence = self._find_peaks_simple(y)
        peak_freq = float(f[peak_idx])

        feat = {
            "f_centers": f.astype(np.float64),
            "radial_curve": Pr.astype(np.float64),
            "slope": float(slope),
            "kurtosis": float(kur),
            "skewness": float(skw),
            "hf_lf_ratio": float(hf_lf),
            "peak_freq": float(peak_freq),
            "peak_prominence": float(peak_prominence),
        }

        if compute_anisotropy:
            H, W = P2.shape
            yy, xx = np.meshgrid(fy, fx, indexing='ij')
            theta = np.arctan2(yy, xx)
            deg = np.deg2rad(15.0)
            h_mask = (np.abs(theta) <= deg) | (np.abs(np.abs(theta) - np.pi) <= deg)
            v_mask = (np.abs(np.abs(theta) - np.pi/2) <= deg)
            Eh = float(np.mean(P2[h_mask]) + self.params.eps)
            Ev = float(np.mean(P2[v_mask]) + self.params.eps)
            feat["anisotropy_ratio"] = float(Eh / Ev)

        if plot_path is not None:
            if image_name is None:
                image_name = "image"
            base_name = os.path.splitext(image_name)[0]
            plot_filename = f"{base_name}_radial_psd.png"
            full_plot_path = os.path.join(plot_path, plot_filename)
            plot_title = f"Radial PSD Curve - {base_name}"
            self.plot_radial_psd(f, Pr, full_plot_path, plot_title)
            feat["plot_saved_to"] = full_plot_path

        return feat

    def plot_radial_psd(self, f_centers: np.ndarray, radial_curve: np.ndarray, 
                       save_path: str, title: str = "Radial PSD Curve") -> None:
        """Plot and save the radial PSD curve."""
        plt.figure(figsize=(10, 6))
        plt.loglog(f_centers, radial_curve, 'b-', linewidth=2, label='Radial PSD')
        plt.xlabel('Frequency (cycles/pixel)', fontsize=12)
        plt.ylabel('Normalized Power', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def add_image_features(self, features: Dict[str, Any]) -> None:
        """Add computed features to the internal list for later aggregation."""
        self.per_image_features.append(features)

    def aggregate_group_features(self, per_image_features: List[Dict[str, Any]] = None) -> Dict[str, Any]: # type: ignore
        """
        Aggregate a list of per-image PSD feature dicts into group-level descriptors.
        If per_image_features is None, uses the internal list.
        Returns:
          - curve_mean[radial_bins], curve_std[radial_bins]
          - slope_mean/std/median
          - kurtosis_mean/std/median
          - skewness_mean/std/median
          - hf_lf_mean/std/median
          - peak_freq_mean/std/median
          - peak_prominence_mean/std/median
          - (optional) anisotropy_mean/std/median
        """
        if per_image_features is None:
            per_image_features = self.per_image_features
            
        if not per_image_features:
            raise ValueError("Empty per_image_features")

        # stack curves (ensure consistent bins)
        f0 = per_image_features[0]["f_centers"]
        curves = []
        for d in per_image_features:
            if not np.allclose(d["f_centers"], f0, rtol=1e-6, atol=1e-8):
                raise ValueError("Inconsistent radial binning across images")
            curves.append(d["radial_curve"])
        curves = np.stack(curves, axis=0)  # [N, K]
        out = {
            "f_centers": f0.astype(np.float64),
            "curve_mean": np.mean(curves, axis=0).astype(np.float64),
            "curve_std": np.std(curves, axis=0).astype(np.float64),
        }

        def agg(key: str):
            vals = np.array([d[key] for d in per_image_features], dtype=np.float64)
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
            out[f"{key}_median"] = float(np.median(vals))

        for k in ["slope", "kurtosis", "skewness", "hf_lf_ratio", "peak_freq", "peak_prominence"]:
            agg(k)
        if "anisotropy_ratio" in per_image_features[0]:
            agg("anisotropy_ratio")

        out["n_images"] = int(len(per_image_features))
        return out

    def load_image(self, path: str) -> np.ndarray:
        """Load image as grayscale float in [0,1]. Supports common formats if Pillow is available."""
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow is required to load images by path in this helper. Install pillow.")
        img = Image.open(path).convert("RGB") # type: ignore
        arr = np.array(img).astype(np.float32) / 255.0
        return self._to_grayscale_float01(arr)

    def process_folder(self, folder: str, compute_anisotropy: bool = False, 
                      plot_path: Optional[str] = None) -> Dict[str, Any]:
        """Process all images in a folder and return aggregated group PSD descriptors."""
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(folder, e)))
        if not files:
            raise FileNotFoundError(f"No images found in: {folder}")

        feats = []
        for fp in files:
            img = self.load_image(fp)
            image_name = os.path.basename(fp)
            d = self.compute_psd_features(
                img, 
                compute_anisotropy=compute_anisotropy,
                plot_path=plot_path,
                image_name=image_name
            )
            feats.append(d)

        group = self.aggregate_group_features(feats)
        group["image_count"] = len(files)
        return group

    def clear_features(self) -> None:
        """Clear the internal list of per-image features."""
        self.per_image_features.clear()


if __name__ == "__main__":
    import cv2
    # Define paths
    img_path = "../tests/test_image_mt.JPG"
    plot_output_dir = "../datafile/test_saving/flot25"

    # Load images
    img = cv2.imread(img_path)
    
    # Extract image name from path
    image_name = os.path.basename(img_path)

    # Create analyzer instance
    analyzer = PSDAnalyzer(PSDParams(radial_bins=32))
    
    # Single image → PSD features with plotting
    feat = analyzer.compute_psd_features(
        img, 
        compute_anisotropy=True,
        plot_path=plot_output_dir,
        image_name=image_name
    )
    print(feat)

    print(f"Plot saved to: {feat.get('plot_saved_to', 'No plot generated')}")
    
    # Example of processing multiple images and aggregating
    # analyzer.add_image_features(feat)
    # group_stats = analyzer.aggregate_group_features()
