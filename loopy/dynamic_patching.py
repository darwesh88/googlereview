from __future__ import annotations

from dataclasses import dataclass

import torch

from loopy.binary_codec_v2 import PAD_BYTE_ID


BOUNDARY_BYTES = set(b" \n\t\r.,!?;:)]}\"'")


@dataclass(frozen=True)
class DynamicPatchEncoding:
    patch_ids: list[list[int]]
    patch_mask: list[float]
    byte_mask: list[list[float]]


def _should_end_patch(value: int, patch_length: int, min_patch_size: int, max_patch_size: int, mode: str) -> bool:
    if patch_length >= max_patch_size:
        return True
    if patch_length < min_patch_size:
        return False
    if mode == "fixed":
        return False
    if mode == "space":
        return value == ord(" ")
    if mode == "boundary":
        return value in BOUNDARY_BYTES
    raise ValueError(f"Unknown dynamic patching mode: {mode}")


def encode_text_to_dynamic_patches(
    text: str,
    max_seq_len: int,
    max_patches: int,
    max_patch_size: int,
    min_patch_size: int = 1,
    patching_mode: str = "boundary",
) -> DynamicPatchEncoding:
    if max_patches <= 0:
        raise ValueError("max_patches must be positive")
    if max_patch_size <= 0:
        raise ValueError("max_patch_size must be positive")
    if min_patch_size <= 0:
        raise ValueError("min_patch_size must be positive")
    if min_patch_size > max_patch_size:
        raise ValueError("min_patch_size cannot exceed max_patch_size")

    byte_values = list(text.encode("utf-8", errors="ignore"))[:max_seq_len]
    patches: list[list[int]] = []
    current: list[int] = []

    for value in byte_values:
        if len(patches) >= max_patches:
            break
        current.append(value)
        if _should_end_patch(value, len(current), min_patch_size, max_patch_size, patching_mode):
            patches.append(current)
            current = []

    if current and len(patches) < max_patches:
        patches.append(current)

    patch_ids: list[list[int]] = []
    patch_mask: list[float] = []
    byte_mask: list[list[float]] = []

    for patch in patches[:max_patches]:
        clipped = patch[:max_patch_size]
        padding = [PAD_BYTE_ID] * (max_patch_size - len(clipped))
        patch_ids.append(clipped + padding)
        patch_mask.append(1.0)
        byte_mask.append([1.0] * len(clipped) + [0.0] * len(padding))

    while len(patch_ids) < max_patches:
        patch_ids.append([PAD_BYTE_ID] * max_patch_size)
        patch_mask.append(0.0)
        byte_mask.append([0.0] * max_patch_size)

    return DynamicPatchEncoding(patch_ids=patch_ids, patch_mask=patch_mask, byte_mask=byte_mask)


def decode_dynamic_patch_ids(patch_ids: torch.Tensor, byte_mask: torch.Tensor | None = None) -> str:
    values = patch_ids.detach().cpu().reshape(-1).tolist()
    if byte_mask is not None:
        mask_values = byte_mask.detach().cpu().reshape(-1).tolist()
        values = [value for value, keep in zip(values, mask_values) if keep > 0.0]
    byte_values = [value for value in values if value != PAD_BYTE_ID]
    return bytes(byte_values).decode("utf-8", errors="ignore")
