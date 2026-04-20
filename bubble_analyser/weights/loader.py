from pathlib import Path

import requests
from tqdm import tqdm


def get_weights_path(filename="mask_rcnn_bubble.h5", download_if_missing=True):
    """Returns the local path to the weights.
    If missing and download_if_missing is True, fetches via terminal (tqdm).
    If False, returns (None, url) so the GUI can handle the download thread.
    """
    # Use the path where this file lives [cite: 50]
    weights_dir = Path(__file__).parent
    local_path = weights_dir / filename

    # Your specific GitHub Release URL
    url = f"https://github.com/ImperialCollegeLondon/bubble_analyser/releases/download/v0.3.0/{filename}"

    if local_path.exists():
        return str(local_path), None

    if not download_if_missing:
        # Return the URL so the PySide6 GUI can download it without freezing
        return None, url

    # Fallback for terminal/script usage
    print(f"Weights not found. Downloading {filename} (~250 MB)...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with (
            open(local_path, "wb") as f,
            tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading Weights") as bar,
        ):
            for data in response.iter_content(chunk_size=16384):  # Larger chunks for efficiency
                f.write(data)
                bar.update(len(data))
        return str(local_path), None
    except Exception as e:
        print(f"Critical error downloading weights: {e}")
        return None, url
