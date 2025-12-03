import os
import numpy as np
import pandas as pd
from PIL import Image

# Where to save processed data
OUT_ROOT = "./data"

# Hugging Face parquet paths (already in your file)
splits = {
    "train": "mnist/train-00000-of-00001.parquet",
    "test": "mnist/test-00000-of-00001.parquet",
}

def load_split(name: str) -> pd.DataFrame:
    """
    Read MNIST split from Hugging Face parquet via fsspec.
    Requires: huggingface_hub, fsspec, pyarrow installed.
    """
    path = "hf://datasets/ylecun/mnist/" + splits[name]
    print(f"Loading {name} split from {path} ...")
    df = pd.read_parquet(path)
    print(f"{name} split loaded: {len(df)} samples")
    print(df.dtypes)
    return df

def ensure_dirs():
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "test"), exist_ok=True)

def save_split(df: pd.DataFrame, split: str, max_count: int = 10000):
    """
    Save first max_count samples of given split into:
      ./data/{split}/{idx}.png
      ./data/{split}_label.txt
    """
    out_dir = os.path.join(OUT_ROOT, split)
    label_path = os.path.join(OUT_ROOT, f"{split}_label.txt")

    os.makedirs(out_dir, exist_ok=True)

    with open(label_path, "w") as lf:
        # use a simple local index 0..max_count-1 so filenames match mnist.py
        for local_idx, (_, row) in enumerate(df.iloc[:max_count].iterrows()):
            # ----- label -----
            label = int(row["label"])
            lf.write(f"{label}\n")

            img = row["image"]

            # ---- decode the image ----
            if isinstance(img, Image.Image):
                # already a PIL image
                pil_img = img.convert("L")

            elif isinstance(img, dict):
                # HF parquet often stores images as dicts, e.g. {"__array__": ..., "shape": ..., "dtype": ...}
                if "__array__" in img:
                    arr = np.array(img["__array__"], dtype=np.uint8)
                    arr = arr.reshape(28, 28)
                    pil_img = Image.fromarray(arr, mode="L")
                elif "bytes" in img:
                    from io import BytesIO
                    pil_img = Image.open(BytesIO(img["bytes"])).convert("L")
                elif "data" in img:
                    arr = np.array(img["data"], dtype=np.uint8)
                    arr = arr.reshape(28, 28)
                    pil_img = Image.fromarray(arr, mode="L")
                else:
                    raise TypeError(f"Unsupported image dict keys: {img.keys()}")

            else:
                # maybe it's already an array or list
                arr = np.array(img, dtype=np.uint8)
                arr = arr.reshape(28, 28)
                pil_img = Image.fromarray(arr, mode="L")

            # Save as 28x28 PNG; we'll pad to 32x32 later in test/train via torchvision.transforms.Pad(2)
            out_path = os.path.join(out_dir, f"{local_idx}.png")
            pil_img.save(out_path)

            if (local_idx + 1) % 1000 == 0:
                print(f"{split}: saved {local_idx + 1} images...")

    print(f"{split}: wrote {min(max_count, len(df))} images and labels to {out_dir} and {label_path}")

def main():
    ensure_dirs()

    # Load both splits
    df_train = load_split("train")
    df_test = load_split("test")

    # Save first 10,000 examples from each
    save_split(df_train, "train", max_count=10000)
    save_split(df_test, "test", max_count=10000)

    print("Done. Check the ./data directory for images and label files.")

if __name__ == "__main__":
    main()
