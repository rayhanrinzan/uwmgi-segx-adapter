from pathlib import Path

import numpy as np
import omegaconf
import pandas as pd

from segx import MULTICLASS_MODE, group_width
from segx.train.engine import SegmentationEngine


class UWMGISegmentationEngine(SegmentationEngine):
    """SegX engine adapter for the converted UW-Madison GI dataset.

    The converted dataset already contains one image PNG, one multiclass mask PNG,
    and one row in ``metadata.csv`` per MRI slice. Newer SegX versions expect
    long-format metadata with ``IMG_NAME``/``MASK_NAME`` and group identifiers, so
    this class translates the existing CSV in memory without rewriting the data.
    """

    @staticmethod
    def read_meta(cfg: omegaconf.DictConfig, logger) -> pd.DataFrame:
        data_dir = Path(cfg.data.data_dir).expanduser().resolve()
        metadata_name = str(cfg.data.get("metadata", "metadata.csv"))
        metadata_path = data_dir / metadata_name

        if not metadata_path.is_file():
            raise FileNotFoundError(f"UW-Madison metadata not found: {metadata_path}")

        source = pd.read_csv(metadata_path)
        required = {
            "FNAME",
            "MASK_organ",
            "ID",
            "case_id",
            "scan_id",
            "slice_id",
        }
        missing = sorted(required.difference(source.columns))
        if missing:
            raise ValueError(
                f"UW-Madison metadata is missing required columns: {missing}. "
                f"Found: {list(source.columns)}"
            )

        groups = list(cfg.data.class_groups)
        if len(groups) != 1:
            raise ValueError(f"UW-Madison adapter expects one class group, found {len(groups)}")

        group = groups[0]
        group_name = str(group["grp_name"])
        group_mode = str(group["mode"])
        if group_mode != MULTICLASS_MODE:
            raise ValueError(
                f"UW-Madison organ masks are multiclass, but config uses mode={group_mode!r}"
            )

        def resolve_from_root(value: str) -> str:
            path = Path(str(value))
            if not path.is_absolute():
                path = data_dir / path
            return str(path.resolve())

        metadata = pd.DataFrame(
            {
                "IMG_NAME": source["FNAME"].map(resolve_from_root),
                "GRP_NAME": group_name,
                "GRP_ID": 0,
                "MASK_NAME": source["MASK_organ"].map(resolve_from_root),
                "MASK_ID": 0,
                "MODE": "mc",
                "ID": source["ID"].astype(str),
                "case_id": source["case_id"].astype(str),
                "scan_id": source["scan_id"].astype(str),
                "slice_id": source["slice_id"].astype(str),
            }
        )

        if "has_foreground" in source.columns:
            metadata["has_foreground"] = source["has_foreground"]

        logger.info(
            "Loaded %d UW-Madison slices from %s and adapted metadata for SegX group %r",
            len(metadata),
            metadata_path,
            group_name,
        )
        return metadata

    @classmethod
    def compute_group_metrics(cls, groups, gt: dict, preds: dict, metrics: list[str]):
        """Use SegX metrics while accounting for implicit multiclass background.

        In the updated SegX group format, multiclass ``classes`` excludes
        background, while masks and model outputs still contain label/channel 0 for
        background. ``group_width`` supplies the correct class count.
        """
        flat: dict[str, float] = {}
        per_group: dict[str, dict] = {}
        micro_per_group = {key: [] for key in metrics}

        for group in groups:
            name = group["grp_name"]
            mode = group["mode"]
            class_names = list(group["classes"])

            if mode == MULTICLASS_MODE:
                num_classes = group_width(group)
                report_names = ["background", *class_names]
            else:
                num_classes = len(class_names)
                report_names = class_names

            group_metrics = cls.compute_stats_metrics(
                gt[name],
                preds[name],
                mode,
                num_classes,
                report_names,
            )
            per_group[name] = group_metrics

            for key, value in group_metrics.items():
                flat[f"{name}_{key}"] = value
            for key in micro_per_group:
                micro_per_group[key].append(group_metrics[key])

        for key, values in micro_per_group.items():
            flat[key] = float(np.mean(values))

        return flat, per_group
