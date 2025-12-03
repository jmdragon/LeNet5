# prototypes_from_digit.py
import os
import numpy as np
from PIL import Image

DIGIT_ROOT = "./data/digits_updated"  # change if your folder name is different

def build_digit_prototypes_from_digit(
    root: str = DIGIT_ROOT,
    num_per_class: int = 200,
    threshold: int = 128
) -> np.ndarray:
    """
    Build 10 prototypes (one per digit 0-9) of size 7x12 (flattened to 84),
    using the DIGIT dataset.

    Steps:
      - For each digit folder 0..9:
        - Take up to `num_per_class` images
        - Convert to grayscale
        - Resize to 12x7
        - Average them to get a mean grayscale bitmap
        - Binarize: pixels < threshold -> +1 (stroke), else -1 (background)
      - Return array of shape (10, 84) with values in {-1, +1}
    """
    prototypes = []

    for digit in range(10):
        digit_dir = os.path.join(root, str(digit))
        files = sorted(
            f for f in os.listdir(digit_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not files:
            raise RuntimeError(f"No image files found in {digit_dir}")

        use_files = files[:num_per_class]
        accum = None

        for fname in use_files:
            path = os.path.join(digit_dir, fname)
            img = Image.open(path).convert("L")   # grayscale

            # Resize to (width=12, height=7) to match F6's 7x12 layout
            img = img.resize((12, 7), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32)  # (7, 12), values 0–255

            if accum is None:
                accum = arr
            else:
                accum += arr

        avg = accum / len(use_files)  # average image

        # Binarize into +1/-1 bitmap (stylized digit)
        # Foreground (stroke) = +1, background = -1
        binarized = np.where(avg < threshold, 1.0, -1.0)  # (7,12)

        prototypes.append(binarized.reshape(-1))  # flatten to length 84

    prototypes = np.stack(prototypes, axis=0).astype(np.float32)  # (10,84)
    return prototypes
