#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# 0 = background
# 1 = large_bowel
# 2 = small_bowel
# 3 = stomach
EVAL_CLASS_MAP = {
    1: 1,  # large_bowel
    3: 2,  # small_bowel
    4: 3,  # stomach
}


def resolve_dataset_path(root_path: str) -> Path:
    """
    Accept either:
      /path/to/.../dataset
    or:
      /path/to/.../folder_containing_dataset
    """
    root = Path(root_path).expanduser().resolve()

    if root.name == "dataset":
        dataset_path = root
    elif (root / "dataset").is_dir():
        dataset_path = root / "dataset"
    else:
        dataset_path = root

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Could not find dataset directory at: {dataset_path}")

    return dataset_path


def pixel_decoder(encoded_pixels, height: int, width: int) -> np.ndarray:
    """
    Decode RLE string into a 2D binary mask.
    This matches the logic from your original repo.
    """
    encoded_pixels = str(encoded_pixels).split()

    starts = [int(encoded_pixels[i]) for i in range(0, len(encoded_pixels), 2)]
    lengths = [int(encoded_pixels[i]) for i in range(1, len(encoded_pixels), 2)]

    mask = np.zeros(height * width, dtype=np.uint8)

    for start, length in zip(starts, lengths):
        start_idx = start - 1
        end_idx = start_idx + length
        mask[start_idx:end_idx] = 1

    return mask.reshape(height, width)


def normalize_scan_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Your original repo used per-slice min-max normalization.
    SegX's default Dataset reads normal images with cv2.imread, so we save uint8 PNGs.
    """
    img = img.astype(np.float32)

    img_min = img.min()
    img_max = img.max()

    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    img = (img * 255.0).round().clip(0, 255).astype(np.uint8)
    return img


def make_safe_id(case_id: str, scan_id: str, slice_id: str) -> str:
    return f"{case_id}__{scan_id}__{slice_id}".replace("/", "_").replace(" ", "_")


def build_segx_dataset(dataset_root: str, out_dir: str, foreground_only: bool = False) -> None:
    dataset_path = resolve_dataset_path(dataset_root)
    out_dir = Path(out_dir).expanduser().resolve()

    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    case_paths = sorted([p for p in dataset_path.iterdir() if p.is_dir()])

    for case_path in tqdm(case_paths, desc="Cases"):
        case_id = case_path.name

        for case_day_path in sorted([p for p in case_path.iterdir() if p.is_dir()]):
            scan_id = case_day_path.name

            scans_path = case_day_path / "scans"
            contour_csv_path = case_day_path / "contours" / "masks_rle.csv"

            if not scans_path.is_dir() or not contour_csv_path.is_file():
                continue

            mask_df = pd.read_csv(contour_csv_path)

            for scan_path in sorted(scans_path.iterdir()):
                if not scan_path.is_file():
                    continue

                slice_parts = scan_path.name.split("_")
                if len(slice_parts) < 2:
                    continue

                slice_id = slice_parts[0] + "_" + slice_parts[1]
                sample_id = make_safe_id(case_id, scan_id, slice_id)

                img = cv2.imread(str(scan_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise FileNotFoundError(f"Could not read image: {scan_path}")

                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                height, width = img.shape[:2]

                label_map = np.zeros((height, width), dtype=np.uint8)

                slice_rows = mask_df[mask_df["SliceID"] == slice_id]

                for _, row in slice_rows.iterrows():
                    mask_type_id = int(row["MaskTypeID"])
                    encoded_pixels = row["EncodedPixels"]

                    if mask_type_id not in EVAL_CLASS_MAP:
                        continue

                    if pd.isna(encoded_pixels) or str(encoded_pixels) == "-1":
                        continue

                    binary_mask = pixel_decoder(encoded_pixels, height, width)
                    class_id = EVAL_CLASS_MAP[mask_type_id]

                    # One class per pixel.
                    label_map[binary_mask == 1] = class_id

                has_foreground = bool((label_map > 0).any())

                if foreground_only and not has_foreground:
                    continue

                img_uint8 = normalize_scan_to_uint8(img)

                # SegX default model/config expects 3-channel images.
                # Since this is grayscale MRI, repeat the channel 3 times.
                img_3ch = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)

                image_rel = f"images/{sample_id}.png"
                mask_rel = f"masks/{sample_id}.png"

                image_out = out_dir / image_rel
                mask_out = out_dir / mask_rel

                cv2.imwrite(str(image_out), img_3ch)
                cv2.imwrite(str(mask_out), label_map)

                rows.append(
                    {
                        "FNAME": image_rel,
                        "MASK_organ": mask_rel,
                        "ID": sample_id,
                        "case_id": case_id,
                        "scan_id": scan_id,
                        "slice_id": slice_id,
                        "has_foreground": int(has_foreground),
                    }
                )

    metadata = pd.DataFrame(rows)

    if metadata.empty:
        raise RuntimeError("No samples were written. Check dataset path and folder structure.")

    metadata_path = out_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    print(f"\nDone.")
    print(f"Wrote {len(metadata)} samples")
    print(f"Images: {images_dir}")
    print(f"Masks:  {masks_dir}")
    print(f"CSV:    {metadata_path}")
    print("\nClass IDs in mask PNGs:")
    print("  0 = background")
    print("  1 = large_bowel")
    print("  2 = small_bowel")
    print("  3 = stomach")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="Path to UW-Madison dataset or folder containing dataset/")
    parser.add_argument("--out-dir", required=True, help="Output folder for SegX-ready data")
    parser.add_argument(
        "--foreground-only",
        action="store_true",
        help="Only keep slices with at least one organ mask. Useful for debugging, but not ideal for final evaluation.",
    )
    args = parser.parse_args()

    build_segx_dataset(
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        foreground_only=args.foreground_only,
    )


if __name__ == "__main__":
    main()