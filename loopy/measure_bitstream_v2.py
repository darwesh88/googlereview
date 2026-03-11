from __future__ import annotations

import argparse
import gzip
import json
import sys
import zlib
from dataclasses import fields
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from loopy.binary_codec_v2 import SemanticBinaryCodec
from loopy.dataset import load_text_samples
from loopy.train_binary_codec_v2 import encode_text_to_patches
from loopy.v2_config import BinaryCodecConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Loopy v2 bitstream size from a trained checkpoint.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing v2_codec.pt")
    parser.add_argument("--data-path", default="", help="Optional override for dataset path")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(run_dir: Path, device: torch.device) -> tuple[SemanticBinaryCodec, BinaryCodecConfig, dict[str, object]]:
    checkpoint_path = run_dir / "v2_codec.pt"
    payload = torch.load(checkpoint_path, map_location=device)
    config_dict = payload["config"]
    allowed = {field.name for field in fields(BinaryCodecConfig)}
    config_kwargs = {key: value for key, value in config_dict.items() if key in allowed}
    config = BinaryCodecConfig(**config_kwargs)
    model = SemanticBinaryCodec(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, payload


def pack_bits(bits: list[int]) -> bytes:
    packed = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        if bit:
            packed[index // 8] |= 1 << (7 - (index % 8))
    return bytes(packed)


def summarize_blob(blob: bytes, raw_text_bytes: int) -> dict[str, float | int]:
    hard_bytes = len(blob)
    zlib_bytes = len(zlib.compress(blob, level=9))
    gzip_bytes = len(gzip.compress(blob, compresslevel=9))
    return {
        "hard_bytes": hard_bytes,
        "hard_bpb": float((hard_bytes * 8) / max(1, raw_text_bytes)),
        "zlib_bytes": zlib_bytes,
        "gzip_bytes": gzip_bytes,
        "zlib_bpb": float((zlib_bytes * 8) / max(1, raw_text_bytes)),
        "gzip_bpb": float((gzip_bytes * 8) / max(1, raw_text_bytes)),
    }


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    run_dir = Path(args.run_dir)
    model, config, payload = load_checkpoint(run_dir, device)

    data_path = args.data_path or config.data_path
    samples = load_text_samples(data_path, dedupe=False)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    all_bits: list[int] = []
    group_bits: list[list[int]] = [[] for _ in config.bit_groups]
    raw_chunks: list[bytes] = []
    total_active_patches = 0

    with torch.no_grad():
        for start in range(0, len(samples), args.batch_size):
            batch_samples = samples[start : start + args.batch_size]
            patch_ids_list = []
            patch_mask_list = []
            raw_bytes_list = []
            for sample in batch_samples:
                patch_ids, patch_mask = encode_text_to_patches(sample, config.max_seq_len, config.patch_size)
                patch_ids_list.append(patch_ids)
                patch_mask_list.append(patch_mask)
                raw_bytes = list(sample.encode("utf-8", errors="ignore"))[: config.max_seq_len]
                raw_bytes_list.append(bytes(raw_bytes))

            patch_ids_tensor = torch.tensor(patch_ids_list, dtype=torch.long, device=device)
            patch_mask_tensor = torch.tensor(patch_mask_list, dtype=torch.float32, device=device)
            forward = model(patch_ids_tensor, patch_mask_tensor)
            bit_values = forward.bit_values.detach().cpu().int()
            patch_mask_cpu = patch_mask_tensor.detach().cpu().bool()

            bit_offsets: list[tuple[int, int]] = []
            cursor = 0
            for group_size in config.bit_groups:
                bit_offsets.append((cursor, cursor + group_size))
                cursor += group_size

            for sample_index, raw_bytes in enumerate(raw_bytes_list):
                active_patch_bits = bit_values[sample_index][patch_mask_cpu[sample_index]]
                active_bits = active_patch_bits.reshape(-1).tolist()
                all_bits.extend(int(bit) for bit in active_bits)
                for patch_bits in active_patch_bits.tolist():
                    for group_index, (start, end) in enumerate(bit_offsets):
                        group_bits[group_index].extend(int(bit) for bit in patch_bits[start:end])
                raw_chunks.append(raw_bytes)
                total_active_patches += int(patch_mask_cpu[sample_index].sum().item())

    raw_blob = b"".join(raw_chunks)
    packed_bits = pack_bits(all_bits)
    packed_group_blobs = [pack_bits(bits) for bits in group_bits]

    raw_text_bytes = len(raw_blob)
    hard_bitstream_bits = len(all_bits)
    hard_bitstream_bytes = len(packed_bits)
    hard_bitstream_bpb = float(hard_bitstream_bits / max(1, raw_text_bytes))
    zlib_bitstream_bytes = len(zlib.compress(packed_bits, level=9))
    gzip_bitstream_bytes = len(gzip.compress(packed_bits, compresslevel=9))
    zlib_raw_bytes = len(zlib.compress(raw_blob, level=9))
    gzip_raw_bytes = len(gzip.compress(raw_blob, compresslevel=9))

    flat_summary = summarize_blob(packed_bits, raw_text_bytes)
    grouped_summaries = []
    grouped_hard_bytes = 0
    grouped_zlib_bytes = 0
    grouped_gzip_bytes = 0
    for group_index, (group_size, bits, blob) in enumerate(zip(config.bit_groups, group_bits, packed_group_blobs)):
        blob_summary = summarize_blob(blob, raw_text_bytes)
        grouped_hard_bytes += int(blob_summary["hard_bytes"])
        grouped_zlib_bytes += int(blob_summary["zlib_bytes"])
        grouped_gzip_bytes += int(blob_summary["gzip_bytes"])
        ones = sum(bits)
        grouped_summaries.append(
            {
                "group_index": group_index,
                "group_bits_per_patch": group_size,
                "total_bits": len(bits),
                "bit_density": float(ones / max(1, len(bits))),
                "hard_bytes": int(blob_summary["hard_bytes"]),
                "hard_bpb": float(blob_summary["hard_bpb"]),
                "zlib_bytes": int(blob_summary["zlib_bytes"]),
                "gzip_bytes": int(blob_summary["gzip_bytes"]),
                "zlib_bpb": float(blob_summary["zlib_bpb"]),
                "gzip_bpb": float(blob_summary["gzip_bpb"]),
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "data_path": str(data_path),
        "device": str(device),
        "samples": len(samples),
        "raw_text_bytes": raw_text_bytes,
        "active_patches": total_active_patches,
        "bits_per_patch": config.total_bits,
        "raw_capacity_bpb": config.raw_capacity_bpb,
        "hard_bitstream_bits": hard_bitstream_bits,
        "hard_bitstream_bytes": hard_bitstream_bytes,
        "hard_bitstream_bpb": hard_bitstream_bpb,
        "zlib_bitstream_bytes": zlib_bitstream_bytes,
        "gzip_bitstream_bytes": gzip_bitstream_bytes,
        "zlib_bitstream_bpb": float((zlib_bitstream_bytes * 8) / max(1, raw_text_bytes)),
        "gzip_bitstream_bpb": float((gzip_bitstream_bytes * 8) / max(1, raw_text_bytes)),
        "flat_packed_summary": flat_summary,
        "grouped_hard_bitstream_bytes": grouped_hard_bytes,
        "grouped_zlib_bitstream_bytes": grouped_zlib_bytes,
        "grouped_gzip_bitstream_bytes": grouped_gzip_bytes,
        "grouped_hard_bitstream_bpb": float((grouped_hard_bytes * 8) / max(1, raw_text_bytes)),
        "grouped_zlib_bitstream_bpb": float((grouped_zlib_bytes * 8) / max(1, raw_text_bytes)),
        "grouped_gzip_bitstream_bpb": float((grouped_gzip_bytes * 8) / max(1, raw_text_bytes)),
        "grouped_bitstreams": grouped_summaries,
        "zlib_raw_bytes": zlib_raw_bytes,
        "gzip_raw_bytes": gzip_raw_bytes,
        "zlib_raw_bpb": float((zlib_raw_bytes * 8) / max(1, raw_text_bytes)),
        "gzip_raw_bpb": float((gzip_raw_bytes * 8) / max(1, raw_text_bytes)),
        "training_best_metrics": payload.get("metrics", {}),
        "notes": [
            "This is a prototype bitstream measurement, not a full entropy-coded production codec.",
            "Sample boundaries are not separately encoded in this measurement.",
            "Grouped packing treats each bit-group stream as a separately stored stream before zlib/gzip.",
            "Use this to compare relative progress between runs, not to make final compression claims."
        ],
    }

    text = json.dumps(summary, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
