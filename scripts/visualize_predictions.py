import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_NAMES = {
    0: "background",
    1: "large_bowel",
    2: "small_bowel",
    3: "stomach",
}

# RGBA-like RGB colors for visualization
CLASS_COLORS = {
    0: np.array([0, 0, 0], dtype=np.uint8),
    1: np.array([255, 80, 80], dtype=np.uint8),    # red-ish
    2: np.array([80, 220, 120], dtype=np.uint8),   # green-ish
    3: np.array([80, 140, 255], dtype=np.uint8),   # blue-ish
}


def make_color_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        out[mask == cls] = color
    return out


def overlay_mask_on_image(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color_mask = make_color_mask(mask)
    overlay = image_rgb.copy().astype(np.float32)
    mask_fg = mask > 0
    overlay[mask_fg] = (
        (1 - alpha) * overlay[mask_fg] + alpha * color_mask[mask_fg].astype(np.float32)
    )
    return overlay.clip(0, 255).astype(np.uint8)


def make_error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Black = correct background
    White = correct foreground class
    Red   = false positive
    Blue  = false negative
    Yellow= wrong organ class (foreground predicted, but wrong class)
    """
    h, w = gt.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    correct_bg = (gt == 0) & (pred == 0)
    correct_fg = (gt == pred) & (gt > 0)
    false_pos = (gt == 0) & (pred > 0)
    false_neg = (gt > 0) & (pred == 0)
    wrong_class = (gt > 0) & (pred > 0) & (gt != pred)

    out[correct_bg] = [0, 0, 0]
    out[correct_fg] = [255, 255, 255]
    out[false_pos] = [255, 0, 0]
    out[false_neg] = [0, 0, 255]
    out[wrong_class] = [255, 255, 0]

    return out


def class_iou(gt: np.ndarray, pred: np.ndarray, cls: int):
    gt_c = gt == cls
    pred_c = pred == cls
    union = np.logical_or(gt_c, pred_c).sum()
    if union == 0:
        return None
    inter = np.logical_and(gt_c, pred_c).sum()
    return inter / union


def foreground_mean_iou(gt: np.ndarray, pred: np.ndarray):
    vals = []
    for cls in [1, 2, 3]:
        iou = class_iou(gt, pred, cls)
        if iou is not None:
            vals.append(iou)
    if not vals:
        return None
    return float(np.mean(vals))


def load_image(data_root: Path, rel_path: str) -> np.ndarray:
    img = cv2.imread(str(data_root / rel_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {data_root / rel_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_panel(
    out_path: Path,
    image_rgb: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    sample_id: str,
    case_id: str,
    scan_id: str,
    slice_id: str,
    fg_iou,
    bucket_name: str,
):
    gt_color = make_color_mask(gt)
    pred_color = make_color_mask(pred)
    err = make_error_map(gt, pred)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 6.2),
    )

    axes[0].imshow(image_rgb)
    axes[0].set_title(
        f"Original MRI\n{image_rgb.shape[1]} × {image_rgb.shape[0]}",
        fontsize=12,
        pad=8,
    )
    axes[0].axis("off")

    axes[1].imshow(gt_color)
    axes[1].set_title(
        f"Ground Truth\n{gt.shape[1]} × {gt.shape[0]}",
        fontsize=12,
        pad=8,
    )
    axes[1].axis("off")

    axes[2].imshow(pred_color)
    axes[2].set_title(
        f"Prediction\n{pred.shape[1]} × {pred.shape[0]}",
        fontsize=12,
        pad=8,
    )
    axes[2].axis("off")

    axes[3].imshow(err)
    axes[3].set_title(
        "Error Map\nRed: FP  |  Blue: FN  |  Yellow: Wrong Class",
        fontsize=11,
        pad=8,
    )
    axes[3].axis("off")

    fg_iou_text = "N/A" if fg_iou is None else f"{fg_iou:.4f}"
    bucket_text = bucket_name.replace("_", " ").title()

    figure_title = (
        f"{bucket_text} Example  |  Foreground Mean IoU: {fg_iou_text}\n"
        f"ID: {sample_id}  |  Case: {case_id}  |  "
        f"Scan: {scan_id}  |  Slice: {slice_id}"
    )

    fig.suptitle(
        figure_title,
        fontsize=11,
        y=0.98,
    )

    # Leave the upper portion of the figure exclusively for the main title.
    fig.tight_layout(
        rect=[0.01, 0.01, 0.99, 0.84],
        w_pad=1.2,
    )

    fig.savefig(
        out_path,
        dpi=160,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file", required=True, help="Path to preds_*.npz")
    parser.add_argument("--data-root", required=True, help="Root of segx_gi dataset")
    parser.add_argument("--out-dir", required=True, help="Where to save visualizations")
    args = parser.parse_args()

    pred_file = Path(args.pred_file)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = data_root / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found at {metadata_csv}")

    print(f"Loading prediction file: {pred_file}")
    data = np.load(pred_file, allow_pickle=True)

    preds = data["preds__organ"]
    gt = data["gt__organ"]
    meta = data["metadata"].item()

    ids = meta["ID"]
    case_ids = meta["case_id"]
    scan_ids = meta["scan_id"]
    slice_ids = meta["slice_id"]

    full_meta = pd.read_csv(metadata_csv)
    id_to_fname = dict(zip(full_meta["ID"], full_meta["FNAME"]))

    rows = []

    for i in range(len(ids)):
        sample_id = ids[i]
        gt_i = gt[i]
        pred_i = preds[i]

        fg_iou = foreground_mean_iou(gt_i, pred_i)

        gt_has_fg = bool((gt_i > 0).any())
        pred_has_fg = bool((pred_i > 0).any())

        rows.append(
            {
                "idx": i,
                "ID": sample_id,
                "case_id": case_ids[i],
                "scan_id": scan_ids[i],
                "slice_id": slice_ids[i],
                "FNAME": id_to_fname.get(sample_id, None),
                "fg_mean_iou": -1.0 if fg_iou is None else fg_iou,
                "gt_has_fg": gt_has_fg,
                "pred_has_fg": pred_has_fg,
            }
        )

    df = pd.DataFrame(rows)

    if df["FNAME"].isna().any():
        missing = df[df["FNAME"].isna()]["ID"].tolist()[:10]
        raise RuntimeError(f"Could not map some IDs to FNAME. Examples: {missing}")

    # Background-only false positives:
    bg_fp = df[(df["gt_has_fg"] == False) & (df["pred_has_fg"] == True)].copy()

    # Foreground cases for best / middle / worst:
    fg_df = df[df["gt_has_fg"] == True].copy().sort_values("fg_mean_iou")

    worst = fg_df.head(5).copy()
    best = fg_df.tail(5).copy()

    # Middle 5 slices
    if len(fg_df) >= 5:
        mid_start = max(0, len(fg_df) // 2 - 2)
        middle = fg_df.iloc[mid_start : mid_start + 5].copy()
    else:
        middle = fg_df.copy()

    bg_fp = bg_fp.head(5).copy()

    selected_groups = [
        ("worst", worst),
        ("middle", middle),
        ("best", best),
        ("background_false_positive", bg_fp),
    ]

    summary_rows = []

    for group_name, group_df in selected_groups:
        group_dir = out_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        for rank, (_, row) in enumerate(group_df.iterrows(), start=1):
            idx = int(row["idx"])
            sample_id = row["ID"]
            image_rgb = load_image(data_root, row["FNAME"])

            save_path = group_dir / f"{rank:02d}_{sample_id}.png"

            save_panel(
                out_path=save_path,
                image_rgb=image_rgb,
                gt=gt[idx],
                pred=preds[idx],
                sample_id=sample_id,
                case_id=row["case_id"],
                scan_id=row["scan_id"],
                slice_id=row["slice_id"],
                fg_iou=None if row["fg_mean_iou"] < 0 else row["fg_mean_iou"],
                bucket_name=group_name,
            )

            summary_rows.append(
                {
                    "bucket": group_name,
                    "rank": rank,
                    "ID": sample_id,
                    "case_id": row["case_id"],
                    "scan_id": row["scan_id"],
                    "slice_id": row["slice_id"],
                    "fg_mean_iou": None if row["fg_mean_iou"] < 0 else row["fg_mean_iou"],
                    "FNAME": row["FNAME"],
                    "saved_to": str(save_path),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\nSaved visualizations to:", out_dir)
    print("Summary CSV:", summary_csv)
    print("\nCounts:")
    for group_name, group_df in selected_groups:
        print(f"  {group_name}: {len(group_df)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
