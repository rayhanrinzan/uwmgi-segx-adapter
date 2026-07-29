from __future__ import annotations

import cv2
import pandas as pd
import solt
import torch
from torch.utils.data import Dataset

from segx import MULTICLASS_MODE


class UWMGISegmentationDataset(Dataset):
    """Dataset adapter for the single multiclass UW-Madison organ mask.

    Updated SegX passes transformed data through SOLT. SOLT returns the singular
    key ``mask`` when there is exactly one mask, while the updated generic SegX
    dataset assumes the plural key ``masks``. UW-Madison has one multiclass mask
    per image, so this adapter uses SOLT's singular-mask path explicitly.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        trf: solt.Stream,
        groups,
        metadata_keys: list[str],
    ) -> None:
        if metadata.empty:
            raise ValueError("Metadata DataFrame cannot be empty")

        self.groups = list(groups)
        if len(self.groups) != 1:
            raise ValueError(
                f"UW-Madison dataset expects one class group, found {len(self.groups)}"
            )

        self.group = self.groups[0]
        self.group_name = str(self.group["grp_name"])
        if self.group["mode"] != MULTICLASS_MODE:
            raise ValueError(
                "UW-Madison organ masks require mode='multiclass', "
                f"found {self.group['mode']!r}"
            )

        self.metadata = metadata
        self.trf = trf
        self.metadata_keys = list(metadata_keys)
        self.data = [(fname, frame) for fname, frame in metadata.groupby("IMG_NAME")]

    def __getitem__(self, idx: int) -> dict:
        image_name, entry = self.data[idx]

        image = cv2.imread(str(image_name), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_name}")

        group_entry = entry.loc[entry["GRP_NAME"] == self.group_name]
        if len(group_entry) != 1:
            raise ValueError(
                f"Expected exactly one {self.group_name!r} mask for {image_name}, "
                f"found {len(group_entry)}"
            )

        mask_name = str(group_entry.iloc[0]["MASK_NAME"])
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_name}")

        transformed = self.trf({"image": image, "mask": mask})

        if "image" not in transformed:
            raise KeyError(
                f"SOLT output did not contain 'image'; keys={list(transformed.keys())}"
            )
        if "mask" not in transformed:
            raise KeyError(
                f"SOLT output did not contain singular 'mask'; keys={list(transformed.keys())}"
            )

        result = {
            "img": transformed["image"],
            f"mask__{self.group_name}": transformed["mask"].long(),
        }

        first_row = entry.iloc[0]
        for key in self.metadata_keys:
            result[key] = str(first_row[key])

        return result

    def __len__(self) -> int:
        return len(self.data)
