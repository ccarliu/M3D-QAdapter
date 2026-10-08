#!/usr/bin/env python3
"""
Data loading utility for preprocessed CT cases.

Provides `load_case` which:
  1. Loads image.npz and masks.npz from a case directory
  2. Center-crops both to a target size (D=240, H=480, W=480)
     - If the volume is smaller than the target along any axis, zero-pad symmetrically
  3. Rescales CT from [0, 1] → [-1, 1]
  4. Returns (image, masks)

Usage:
    from data_loader import load_case
    image, masks = load_case("/path/to/preprocessed/BDMAP_00000001")
    # image: np.float32, shape (D, H, W),       range [-1, 1]
    # masks: np.uint8,   shape (10, D, H, W),   binary
"""

import os
import numpy as np

# ──────────────────────────── Config ─────────────────────────────────────
PREPROCESSED_DIR = "/apdcephfs/private_carlohliu/abdomen/preprocessed"

# Target crop size: (D, H, W)
TARGET_D = 240
TARGET_H = 480
TARGET_W = 480

# Channel order in masks (for reference)
CHANNEL_NAMES = [
    "colon",            # 0
    "colon_lesion",     # 1
    "kidney_left",      # 2
    "kidney_right",     # 3
    "kidney_lesion",    # 4
    "liver",            # 5
    "liver_lesion",     # 6
    "pancreas",         # 7
    "pancreatic_lesion",# 8
    "spleen",           # 9
]


def center_crop_or_pad_3d(volume, target_shape):
    """
    Center-crop (or zero-pad if smaller) a 3D array to target_shape.

    Parameters
    ----------
    volume : np.ndarray, shape (D, H, W) or (C, D, H, W)
        Input volume. If 4D, crop/pad is applied on the last 3 dims.
    target_shape : tuple of int
        (target_D, target_H, target_W)

    Returns
    -------
    np.ndarray with spatial dims == target_shape
    """
    is_4d = volume.ndim == 4
    if is_4d:
        C = volume.shape[0]
        spatial = volume.shape[1:]
    else:
        spatial = volume.shape

    tD, tH, tW = target_shape
    sD, sH, sW = spatial

    # Allocate output
    if is_4d:
        out = np.zeros((C, tD, tH, tW), dtype=volume.dtype)
    else:
        out = np.zeros((tD, tH, tW), dtype=volume.dtype)

    # For each axis, compute source slice and destination slice
    def _slices(src_size, tgt_size):
        if src_size >= tgt_size:
            # Center crop: take middle tgt_size from source
            start = (src_size - tgt_size) // 2
            src_slice = slice(start, start + tgt_size)
            dst_slice = slice(0, tgt_size)
        else:
            # Zero pad: place source in center of target
            pad = (tgt_size - src_size) // 2
            src_slice = slice(0, src_size)
            dst_slice = slice(pad, pad + src_size)
        return src_slice, dst_slice

    sd, dd = _slices(sD, tD)
    sh, dh = _slices(sH, tH)
    sw, dw = _slices(sW, tW)

    if is_4d:
        out[:, dd, dh, dw] = volume[:, sd, sh, sw]
    else:
        out[dd, dh, dw] = volume[sd, sh, sw]

    return out


def _resolve_case_path(case_path):
    """Resolve case path to absolute directory path."""
    if not os.path.isabs(case_path) or not os.path.isdir(case_path):
        case_path = os.path.join(PREPROCESSED_DIR, case_path)
    return case_path


def load_image(case_path, target_shape=(TARGET_D, TARGET_H, TARGET_W)):
    """
    Load only the image from a preprocessed case, center-crop/pad, and normalize.

    Parameters
    ----------
    case_path : str
        Path to a case directory containing image.npz.
    target_shape : tuple of int, optional
        (D, H, W) target spatial size. Default (240, 480, 480).

    Returns
    -------
    image : np.ndarray, float32, shape (D, H, W), range normalized
    """
    case_path = _resolve_case_path(case_path)
    img_path = os.path.join(case_path, "image.npz")

    image = np.load(img_path, allow_pickle=True)["data"].transpose(2, 0, 1)  # float32, (D, H, W)
    image = center_crop_or_pad_3d(image, target_shape)   # (D, H, W)
    image = image / 1000

    return image


def load_masks(case_path):
    """
    Load only the masks (tokens) from a preprocessed case.

    Parameters
    ----------
    case_path : str
        Path to a case directory containing tokens.npz.

    Returns
    -------
    masks : np.ndarray, uint8, shape (10, N) or similar, binary
    """
    case_path = _resolve_case_path(case_path)
    msk_path = os.path.join(case_path, "tokens.npz")

    masks = np.load(msk_path, allow_pickle=True)["data"]
    return masks


def load_case(case_path, target_shape=(TARGET_D, TARGET_H, TARGET_W)):
    """
    Load a preprocessed case, center-crop/pad, and normalize.
    (Legacy function that loads both image and masks together.)

    Parameters
    ----------
    case_path : str
        Path to a case directory containing image.npz and tokens.npz.
        Can be either a full path or just a case ID (e.g. "BDMAP_00000001"),
        in which case PREPROCESSED_DIR is prepended.
    target_shape : tuple of int, optional
        (D, H, W) target spatial size. Default (240, 480, 480).

    Returns
    -------
    image : np.ndarray, float32, shape (D, H, W), range normalized
    masks : np.ndarray, uint8,   shape (10, N) or similar, binary
    """
    image = load_image(case_path, target_shape)
    masks = load_masks(case_path)

    return image, masks


# ──────────────────────────── Quick test ─────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        case = sys.argv[1]
    else:
        # Pick the first case in PREPROCESSED_DIR
        cases = sorted(os.listdir(PREPROCESSED_DIR))
        if not cases:
            print("No cases found in", PREPROCESSED_DIR)
            sys.exit(1)
        case = cases[0]

    print(f"Loading case: {case}")
    img, msk = load_case(case)

    print(f"  Image: shape={img.shape}, dtype={img.dtype}, "
          f"min={img.min():.4f}, max={img.max():.4f}")
    print(f"  Masks: shape={msk.shape}, dtype={msk.dtype}, "
          f"unique values={np.unique(msk)}")

    for i, name in enumerate(CHANNEL_NAMES):
        nz = np.count_nonzero(msk[i])
        print(f"    [{i}] {name:<22} nonzero={nz:>10}")
