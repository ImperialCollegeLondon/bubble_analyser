import pandas as pd
import cv2
import numpy as np
import h5py # type: ignore
from typing import cast
from pathlib import Path
from data_aggregation.get_psd_features import PSDAnalyzer
from data_aggregation.get_ellipse_data import MTImageProcessor

class DataGroup:
    def __init__(self, excel_path: Path, img_folder_path = cast(Path, None)) -> None:
        self.excel_path = excel_path
        self.img_folder_path = img_folder_path

        self.mt_img_list = []
        self.rbg_img_list = []
        self.real_eqd_list = []

    def get_image_mt_list(self) -> tuple[list, list]:
        # Read the specific sheet "data_group_1"
        df = pd.read_excel(self.excel_path, sheet_name='data_group_1')

        # Extract unique values from the 'image name' column
        self.rbg_img_list = df['image name'].unique().tolist()

        # Extract real equivalent diameter 
        self.real_eqd_list = df['equivalent_diameter(px)'].tolist()
        self.real_d32 = df['D32(Sauter Mean Diameter/px)'].iloc[0]
        print(self.real_d32)

        # Modify each image name: remove .JPG and add _mt.png
        # self.mt_img_list = []
        for name in self.rbg_img_list:
            # Remove .JPG extension and add _mt.png
            mt_img_names = name.replace('.JPG', '_mt.png')
            mt_img_names = f"{self.img_folder_path}/{mt_img_names}"
            self.mt_img_list.append(mt_img_names)
        return self.rbg_img_list, self.mt_img_list
    
    def get_ground_truth(self):
        # Read the specific sheet "data_group_1"
        df = pd.read_excel(self.excel_path, sheet_name='data_group_1')
        return df

    def get_ellipse_data(self, binary_image):
        # Process the image with filtering parameters
        # Adjust min_area and min_contour_length based on your image characteristics
        processor = MTImageProcessor(px2mm=1, resample=1, min_area=100, min_contour_length=15)
        ellipses, properties = processor.process_binary_image(binary_image)
        
        return ellipses, properties

    def get_normalized_histogram(self, input_hist, n_bins: int = 64, log_scale: bool = True):
        """
        Convert the real_eqd_list into a normalized histogram (sum = 1).

        Parameters
        ----------
        n_bins : int
            Number of histogram bins (default: 64).
        log_scale : bool
            If True, use log-spaced bins (better for bubble sizes).
            If False, use linear-spaced bins.

        Returns
        -------
        hist : np.ndarray
            Normalized histogram of length n_bins (density, sum = 1).
        bin_edges : np.ndarray
            The edges of the bins (length n_bins+1).
        """
        if not input_hist or len(input_hist) == 0:
            raise ValueError("input_hist is empty.")

        values = np.array(input_hist, dtype=np.float32)

        # Choose binning strategy
        if log_scale:
            min_val = max(values.min(), 1e-3)  # avoid log(0)
            max_val = values.max()
            bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), n_bins + 1)
        else:
            bin_edges = np.linspace(values.min(), values.max(), n_bins + 1)

        # Compute histogram (counts)
        hist, _ = np.histogram(values, bins=bin_edges)

        # Normalize to density (sum = 1)
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist = hist / hist.sum()

        return hist, bin_edges

    def get_psd_features(self, binary_image):
        psd_analyzer = PSDAnalyzer()
        psd_features = psd_analyzer.compute_psd_features(binary_image)
        return psd_features

    def get_all_data(self):
        self.all_data_images = []
        self.all_data_target = {}
        self.all_data_meta = {}

        # meta_data
        n_bubbles = 0
        n_images = 0
        # meta_descriptor
        eqd_list = []
        area_list = []
        peri_list = []
        ecc_list = []

        for img_path in self.mt_img_list:
            n_images += 1
            img_dict = {}
            img_dict['img_path'] = img_path

            try:
                binary_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            except:
                print(f"Could not load image: {img_path}")
                continue
            
            # get ellipse data
            _, ellipse_data = self.get_ellipse_data(binary_image)
            n_bubbles += len(ellipse_data)
            # get psd data
            psd_feature = self.get_psd_features(binary_image)

            img_dict['ellipse_data'] = ellipse_data
            img_dict['psd_feature'] = psd_feature

            self.all_data_images.append(img_dict)

            for properties in ellipse_data:
                eqd_list.append(properties['equivalent_diameter_px'])
                area_list.append(properties['area_px2'])
                peri_list.append(properties['perimeter_px'])
                ecc_list.append(properties['eccentricity'])

        # get real_eqdiam histogram
        eqdiam_hist, bin_edges = self.get_normalized_histogram(self.real_eqd_list)
        # get real d32
        real_d32 = self.real_d32
        self.all_data_target['target_eqdiam_hist'] = eqdiam_hist
        self.all_data_target['target_eqdiam_bin_edges'] = bin_edges
        self.all_data_target['target_d32'] = real_d32


        eqd_hist_input, eqd_bin_edges = self.get_normalized_histogram(eqd_list)
        area_hist_input, area_bin_edges = self.get_normalized_histogram(area_list)
        peri_hist_input, peri_bin_edges = self.get_normalized_histogram(peri_list)
        ecc_hist_input, ecc_bin_edges = self.get_normalized_histogram(ecc_list)
        self.all_data_meta['n_bubbles'] = n_bubbles
        self.all_data_meta['n_images'] = n_images
        self.all_data_meta['eqd_hist_input'] = eqd_hist_input
        self.all_data_meta['area_hist_input'] = area_hist_input
        self.all_data_meta['peri_hist_input'] = peri_hist_input
        self.all_data_meta['ecc_hist_input'] = ecc_hist_input
        self.all_data_meta['eqd_bin_edges'] = eqd_bin_edges
        self.all_data_meta['area_bin_edges'] = area_bin_edges
        self.all_data_meta['peri_bin_edges'] = peri_bin_edges
        self.all_data_meta['ecc_bin_edges'] = ecc_bin_edges

    def _build_per_bubble_matrix(self) -> np.ndarray:
        """
        Stack all bubble properties from all images into a single matrix [B, 5]:
        [area_px2, eq_diam_px, eccentricity, solidity, circularity]
        """
        rows = []
        for row in getattr(self, "all_data_images", []):
            ellipse_list = row.get("ellipse_data", [])
            for props in ellipse_list:
                rows.append([
                    float(props.get("area_px2", np.nan)),
                    float(props.get("equivalent_diameter_px", np.nan)),
                    float(props.get("eccentricity", np.nan)),
                    float(props.get("perimeter_px", np.nan)),
                ])
        if not rows:
            return np.zeros((0, 5), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def _build_per_image_matrix(self, expect_K: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Build per-image matrix [M, 6+K] with columns:
        [psd_slope, psd_kurtosis, psd_skewness, hf_lf, peak_frequency, peak_prominence, radial_psd(K)]
        Returns:
        X_img : float32 [M, 6+K]
        f_centers : float64 [K]  (frequency bin centers; NaNs if unavailable)
        """
        rows = []
        f_centers_ref = None
        for row in getattr(self, "all_data_images", []):
            feat = row.get("psd_feature", {}) or {}

            radial = np.asarray(feat.get("radial_curve", []), dtype=np.float32)
            if radial.size == 0:
                raise ValueError(f"Missing 'radial_curve' in psd_feature for image {row.get('img_path','?')}")
            if expect_K is not None and radial.size != expect_K:
                raise ValueError(f"Inconsistent K. Expected {expect_K}, got {radial.size} for {row.get('img_path','?')}")
            K = radial.size

            slope      = float(feat.get("slope", np.nan))
            kurt       = float(feat.get("kurtosis", np.nan))
            skew       = float(feat.get("skewness", np.nan))
            hf_lf      = float(feat.get("hf_lf_ratio", np.nan))
            peak_freq  = float(feat.get("peak_freq", np.nan))
            peak_prom  = float(feat.get("peak_prominence", np.nan))

            head = np.array([slope, kurt, skew, hf_lf, peak_freq, peak_prom], dtype=np.float32)
            rows.append(np.concatenate([head, radial.astype(np.float32)], axis=0))

            f_centers = feat.get("f_centers", None)
            if f_centers is not None:
                f_centers = np.asarray(f_centers, dtype=np.float64)
                if f_centers_ref is None:
                    f_centers_ref = f_centers
                else:
                    if f_centers_ref.shape != f_centers.shape or not np.allclose(f_centers_ref, f_centers, rtol=1e-6, atol=1e-12):
                        raise ValueError("Inconsistent PSD frequency bins (f_centers) across images.")

        if not rows:
            # return empty with the expected column count if provided
            cols = (6 + (expect_K or 0))
            return np.zeros((0, cols), dtype=np.float32), np.full((expect_K or 0,), np.nan, dtype=np.float64)

        X_img = np.vstack(rows).astype(np.float32)
        if f_centers_ref is None:
            f_centers_ref = np.full((X_img.shape[1] - 6,), np.nan, dtype=np.float64)

        return X_img, f_centers_ref

    def _require_keys(self, d: dict, keys: list[str], where: str):
        missing = [k for k in keys if k not in d]
        if missing:
            raise RuntimeError(f"Missing keys in {where}: {missing}")

    def save_group_to_hdf5(self, out_path: Path) -> None:
        """
        Save the group to an HDF5 file with:
        /meta/...
        /bubbles/features           [B,5]
        /images/features            [M,6+K]
        /images/f_centers           [K]
        /descriptors/...            (input histograms + edges)
        /target/...
        Uses gzip compression and chunking where appropriate.
        """
        # --- sanity: ensure get_all_data() was run
        if not hasattr(self, "all_data_images") or not hasattr(self, "all_data_meta") or not hasattr(self, "all_data_target"):
            raise RuntimeError("Run get_all_data() before saving.")

        # --- gather matrices
        X_bub = self._build_per_bubble_matrix()  # [B,5]

        first_psd = None
        for row in self.all_data_images:
            pf = row.get("psd_feature", {})
            if pf and "radial_curve" in pf:
                first_psd = pf
                break
        K = len(first_psd["radial_curve"]) if first_psd is not None else 0
        X_img, f_centers = self._build_per_image_matrix(expect_K=K)  # [M,6+K], [K]

        # --- targets (DICT now)
        tgt = self.all_data_target
        self._require_keys(tgt, ["target_eqdiam_hist", "target_eqdiam_bin_edges", "target_d32"], "all_data_target")
        target_eqdiam_hist  = np.asarray(tgt["target_eqdiam_hist"], dtype=np.float32)
        target_eqdiam_edges = np.asarray(tgt["target_eqdiam_bin_edges"], dtype=np.float64)
        target_d32          = float(tgt["target_d32"])

        # --- meta + inputs (DICT now)
        meta = self.all_data_meta
        self._require_keys(
            meta,
            ["n_bubbles", "n_images",
            "eqd_hist_input", "area_hist_input", "peri_hist_input", "ecc_hist_input",
            "eqd_bin_edges", "area_bin_edges", "peri_bin_edges", "ecc_bin_edges"],
            "all_data_meta"
        )
        n_bubbles = int(meta["n_bubbles"])
        n_images  = int(meta["n_images"])

        eqd_hist_input  = np.asarray(meta["eqd_hist_input"],  dtype=np.float32)
        area_hist_input = np.asarray(meta["area_hist_input"], dtype=np.float32)
        peri_hist_input = np.asarray(meta["peri_hist_input"], dtype=np.float32)
        ecc_hist_input  = np.asarray(meta["ecc_hist_input"],  dtype=np.float32)

        eqd_edges  = np.asarray(meta["eqd_bin_edges"],  dtype=np.float64)
        area_edges = np.asarray(meta["area_bin_edges"], dtype=np.float64)
        peri_edges = np.asarray(meta["peri_bin_edges"], dtype=np.float64)
        ecc_edges  = np.asarray(meta["ecc_bin_edges"],  dtype=np.float64)


        # --- write to HDF5
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(out_path, "w") as f:
            # meta
            g_meta = f.create_group("meta")
            g_meta.create_dataset("n_bubbles_total", data=np.int32(n_bubbles))
            g_meta.create_dataset("n_images", data=np.int32(n_images))

            # bubbles
            g_bub = f.create_group("bubbles")
            g_bub.create_dataset("features", data=X_bub, compression="gzip", shuffle=True, chunks=True)

            # images
            g_img = f.create_group("images")
            g_img.create_dataset("features", data=X_img, compression="gzip", shuffle=True, chunks=True)
            if K > 0:
                g_img.create_dataset("f_centers", data=f_centers)

            # descriptors (input histograms)
            g_desc = f.create_group("descriptors")
            g_desc.create_dataset("eqd_hist",  data=eqd_hist_input,  compression="gzip")
            g_desc.create_dataset("area_hist", data=area_hist_input, compression="gzip")
            g_desc.create_dataset("peri_hist", data=peri_hist_input, compression="gzip")
            g_desc.create_dataset("ecc_hist",  data=ecc_hist_input,  compression="gzip")

            # bin edges as attributes
            g_desc.attrs["eqd_bin_edges_px"]   = eqd_edges
            g_desc.attrs["area_bin_edges_px2"] = area_edges
            g_desc.attrs["peri_bin_edges_px"]  = peri_edges
            g_desc.attrs["ecc_bin_edges"]      = ecc_edges

            # targets
            g_tgt = f.create_group("target")
            g_tgt.create_dataset("bsd_hist_px", data=target_eqdiam_hist, compression="gzip")
            g_tgt.create_dataset("d32_px",      data=np.float32(target_d32))
            g_tgt.attrs["eqd_bin_edges_px"] = target_eqdiam_edges

        print(f"[OK] Saved group to HDF5: {out_path}  (B={X_bub.shape[0]}, M={X_img.shape[0]}, K={K})")

def export_hdf5_to_excel_detailed(h5_path: Path, out_path: Path) -> None:
    """
    Read a saved HDF5 group file and export:
      - Summary (property, value)
      - Target (property, value) including BSD hist and its bin edges
      - Descriptors (property, value) including input histograms + bin edges
      - PerImage table (one row per image: 6 scalars + K radial PSD bins)
      - Frequencies (f_centers) if available

    Parameters
    ----------
    h5_path : Path
        Path to the HDF5 produced by save_group_to_hdf5().
    out_path : Path
        Destination .xlsx path.
    """
    from pathlib import Path

    h5_path = Path(h5_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def kv_rows(d: dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame({"property": list(d.keys()), "value": list(d.values())})

    with h5py.File(h5_path, "r") as f:
        # ---------- Summary (scalars) ----------
        summary = {
            "n_bubbles_total": int(f["meta/n_bubbles_total"][()]) if "n_bubbles_total" in f["meta"] else np.nan,
            "n_images": int(f["meta/n_images"][()]) if "n_images" in f["meta"] else np.nan,
        }
        df_summary = kv_rows(summary)

        # ---------- Target (hist + edges) ----------
        target_rows = {}
        if "bsd_hist_px" in f["target"]:
            t_hist = f["target/bsd_hist_px"][:]
            for i, v in enumerate(t_hist):
                target_rows[f"target_bsd_hist[{i}]"] = float(v)
        t_edges = f["target"].attrs.get("eqd_bin_edges_px", None)
        if t_edges is not None:
            for i, v in enumerate(t_edges):
                target_rows[f"target_bin_edge[{i}]"] = float(v)
        if "d32_px" in f["target"]:
            target_rows["target_d32_px"] = float(f["target/d32_px"][()])
        df_target = kv_rows(target_rows)

        # ---------- Descriptors (input histograms + edges) ----------
        desc_rows = {}
        for name in ["eqd_hist", "area_hist", "peri_hist", "ecc_hist"]:
            if name in f["descriptors"]:
                arr = f[f"descriptors/{name}"][:]
                for i, v in enumerate(arr):
                    desc_rows[f"{name}[{i}]"] = float(v)

        # edges saved as attributes on /descriptors
        edges_map = {
            "eqd_bin_edges_px": "eqd_edges_px",
            "area_bin_edges_px2": "area_edges_px2",
            "peri_bin_edges_px": "peri_edges_px",
            "ecc_bin_edges": "ecc_edges",
        }
        for attr_key, pretty in edges_map.items():
            edges = f["descriptors"].attrs.get(attr_key, None)
            if edges is not None:
                for i, v in enumerate(edges):
                    desc_rows[f"{pretty}[{i}]"] = float(v)

        df_descriptors = kv_rows(desc_rows)

        # ---------- Per-image matrix ----------
        df_images = pd.DataFrame()
        if "features" in f["images"]:
            X_img = f["images/features"][:]  # shape [M, 6+K]
            M, cols = X_img.shape if X_img.ndim == 2 else (0, 0)
            if M > 0:
                # first 6 columns are scalars, rest are radial bins
                if cols < 6:
                    raise ValueError(f"/images/features has {cols} columns, expected at least 6.")
                K = cols - 6
                head_cols = ["slope", "kurtosis", "skewness", "hf_lf", "peak_frequency", "peak_prominence"]
                radial_cols = [f"radial_{i}" for i in range(K)]
                df_images = pd.DataFrame(X_img, columns=head_cols + radial_cols)

        # ---------- Per-bubble matrix ---------
        df_bubbles = pd.DataFrame()
        if "features" in f["bubbles"]:
            X_bub = f["bubbles/features"][:]  # shape [B, 4]
            B, cols = X_bub.shape if X_bub.ndim == 2 else (0, 0)
            if B > 0:
                # first 5 columns are scalars
                if cols < 4:
                    raise ValueError(f"/bubbles/features has {cols} columns, expected at least 4.")
                head_cols = ["area_px2", "eq_diam_px", "eccentricity", "perimeter_px"]
                df_bubbles = pd.DataFrame(X_bub, columns=head_cols)


        # ---------- Frequencies (f_centers) ----------
        df_freq = pd.DataFrame()
        if "f_centers" in f["images"]:
            fcent = f["images/f_centers"][:]
            df_freq = pd.DataFrame({"bin": np.arange(len(fcent), dtype=int), "f_center": fcent})

    # ---------- Write to Excel (multiple sheets) ----------
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_target.to_excel(writer, sheet_name="Target", index=False)
        df_descriptors.to_excel(writer, sheet_name="Descriptors", index=False)
        if not df_images.empty:
            df_images.to_excel(writer, sheet_name="PerImage", index=False)
        if not df_freq.empty:
            df_freq.to_excel(writer, sheet_name="Frequencies", index=False)
        if not df_bubbles.empty:
            df_bubbles.to_excel(writer, sheet_name="PerBubble", index=False)

        # a little nicer column width for the two-column sheets
        workbook = writer.book
        for sheet in ["Summary", "Target", "Descriptors"]:
            if sheet in writer.sheets:
                ws = writer.sheets[sheet]
                ws.set_column(0, 0, 36)  # property
                ws.set_column(1, 1, 22)  # value

    print(f"[OK] Wrote Excel: {out_path}")

if __name__ == "__main__":
    excel_path = Path("C:/Users/Yiyang/OneDrive/Nov_2025_Conference/coding_part/processed_data_from_ba/OK_Stator_2_phases_DF250_Test 1_2 DF250_4PPM/analyse_result_ba.xlsx")
    img_folder_path = Path("C:/Users/Yiyang/OneDrive/Nov_2025_Conference/coding_part/processed_data_from_ba/OK_Stator_2_phases_DF250_Test 1_2 DF250_4PPM")
    
    data_group = DataGroup(excel_path, img_folder_path)
    _, _ = data_group.get_image_mt_list()
    data_group.get_all_data()
    data_group.save_group_to_hdf5(Path("analyse_result_ba.h5"))

    export_hdf5_to_excel_detailed(Path("analyse_result_ba.h5"), Path("analyse_result_ba.xlsx"))



