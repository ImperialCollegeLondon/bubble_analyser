from typing import cast

import cv2
import numpy as np

from bubble_analyser.methods.watershed_methods import TestWatershed

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    from pathlib import Path
    
    # Define paths dynamically based on current file location
    current_dir = Path(__file__).parent
    img_grey_path = str(current_dir / "test_image_grey.JPG")
    img_rgb_path = str(current_dir / "test_image_rgb.JPG")

    # Change to your desired output location
    background_path = None  # Change if you have a background image

    # Load images
    img_rgb = cv2.imread(img_rgb_path)
    if img_rgb is None:
        raise ValueError(f"Error: Could not load image at {img_rgb_path}")

    img_grey = cv2.imread(img_grey_path, cv2.IMREAD_GRAYSCALE)

    # Load optional background image
    bknd_img = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE) if background_path else None

    params: dict[str, float | int | str] = {
        "resample": 0.5,
        "max_thresh": 0.9,
        "min_thresh": 0.1,
        "step_size": 0.1,
        "element_size": 5,
        "connectivity": 4,
        "threshold_value": 0.15,
        "ksize": 3,
        "if_gaussianblur": "True",
    }
    # Run Iterative Watershed Segmentation without bknd img
    normal_watershed = TestWatershed(cast(dict[str, float | int], params))

    normal_watershed.initialize_processing(
        cast(dict[str, float | int], params),
        img_grey,  # type: ignore
        img_rgb,  # type: ignore
        if_bknd_img=False,
    )

    segmented_img, labels_watershed, _ = normal_watershed.get_results_img()
    img_grey_thresh = normal_watershed.img_grey_thresholded
    dist_transform = normal_watershed.img_grey_dt
    ch_labels = normal_watershed.labels_watershed
    img_morph = normal_watershed.img_grey_morph
    img_morph_eroded = normal_watershed.img_grey_morph_eroded
    img_grey_dt_thresh = normal_watershed.img_grey_dt_thresh

    out_dir = current_dir / "with_grng_DT" / "without_gaussianblur"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_grey_thresh_path = out_dir / "grey_thresh.JPG"
    img_dt_path = out_dir / "dt.JPG"
    img_dt_thresh_save_path = out_dir / "dt_thresh.JPG"
    img_morph_save_path = out_dir / "mt.JPG"
    img_morph_eroded_save_path = out_dir / "mt_eroded.JPG"
    img_segmented_save_path = out_dir / "segmented.JPG"
    grad_img_save_path = out_dir / "grad.JPG"
    output_path = out_dir / "plot_together.JPG"
    np.save(out_dir / "test_labels_watershed.npy", labels_watershed)
    cv2.imwrite(str(img_grey_thresh_path), img_grey_thresh.astype(np.uint8))
    cv2.imwrite(str(img_dt_path), dist_transform)
    cv2.imwrite(str(img_morph_save_path), img_morph * 255)
    cv2.imwrite(str(img_segmented_save_path), segmented_img)
    cv2.imwrite(str(img_dt_thresh_save_path), img_grey_dt_thresh * 255)
    # cv2.imwrite(str(img_morph_eroded_save_path), img_morph_eroded*255)

    # Save and display results
    plt.figure(figsize=(20, 15))  # Increased from (10, 5) to (20, 15)
    plt.subplot(331)
    plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    plt.title("Original Image", fontsize=14)

    plt.subplot(332)
    plt.imshow(img_morph, cmap="jet")
    plt.title("img_morph", fontsize=14)

    plt.subplot(333)
    plt.imshow(img_grey_thresh, cmap="jet")
    plt.title("img_grey_thresh", fontsize=14)

    plt.subplot(334)
    plt.imshow(dist_transform, cmap="gray")
    plt.title("Distance Transform", fontsize=14)

    plt.subplot(335)
    plt.imshow(img_grey_dt_thresh, cmap="jet")
    plt.title("img_grey_dt_thresh", fontsize=14)

    plt.subplot(336)
    plt.imshow(ch_labels, cmap="gray")
    plt.title("ch_labels", fontsize=14)

    plt.subplot(337)
    plt.imshow(segmented_img, cmap="jet")
    plt.title("Segmented Image", fontsize=14)

    plt.subplot(338)
    plt.imshow(normal_watershed.grad_img_rgb, cmap="gray")
    plt.title("grad_img_rgb", fontsize=14)

    # Adjust layout and spacing
    plt.tight_layout(pad=2.0)

    # Save with high resolution and quality
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    print(f"Segmentation completed! Output saved at: {output_path}")
