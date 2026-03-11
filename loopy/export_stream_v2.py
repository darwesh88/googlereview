from __future__ import annotations

import argparse
import json
import sys
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
    parser = argparse.ArgumentParser(description="Export Loopy v2 learned streams for downstream LM experiments.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing v2_codec.pt")
    parser.add_argument("--data-path", default="", help="Optional override for dataset path")
    parser.add_argument("--output-dir", required=True, help="Directory to write exported corpora")
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


def bits_to_int(bits: list[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def export_group_stream(
    model: SemanticBinaryCodec,
    config: BinaryCodecConfig,
    samples: list[str],
    device: torch.device,
    batch_size: int,
) -> list[str]:
    lines: list[str] = []
    bit_offsets: list[tuple[int, int]] = []
    cursor = 0
    for group_size in config.bit_groups:
        bit_offsets.append((cursor, cursor + group_size))
        cursor += group_size

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            patch_ids_list = []
            patch_mask_list = []
            for sample in batch_samples:
                patch_ids, patch_mask = encode_text_to_patches(sample, config.max_seq_len, config.patch_size)
                patch_ids_list.append(patch_ids)
                patch_mask_list.append(patch_mask)

            patch_ids_tensor = torch.tensor(patch_ids_list, dtype=torch.long, device=device)
            patch_mask_tensor = torch.tensor(patch_mask_list, dtype=torch.float32, device=device)
            forward = model(patch_ids_tensor, patch_mask_tensor)
            bit_values = forward.bit_values.detach().cpu().int()
            patch_mask_cpu = patch_mask_tensor.detach().cpu().bool()

            for sample_index in range(len(batch_samples)):
                tokens: list[str] = []
                active_patch_bits = bit_values[sample_index][patch_mask_cpu[sample_index]]
                for patch_bits in active_patch_bits.tolist():
                    tokens.append("<p>")
                    for group_index, (group_start, group_end) in enumerate(bit_offsets):
                        group_value = bits_to_int(patch_bits[group_start:group_end])
                        tokens.append(f"<g{group_index}:{group_value}>")
                lines.append(" ".join(tokens))
    return lines


def export_raw_byte_stream(config: BinaryCodecConfig, samples: list[str]) -> list[str]:
    lines: list[str] = []
    for sample in samples:
        byte_values = list(sample.encode("utf-8", errors="ignore"))[: config.max_seq_len]
        tokens: list[str] = []
        for patch_start in range(0, len(byte_values), config.patch_size):
            tokens.append("<p>")
            patch = byte_values[patch_start : patch_start + config.patch_size]
            for value in patch:
                tokens.append(f"<b:{value}>")
        lines.append(" ".join(tokens))
    return lines


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    model, config, payload = load_checkpoint(run_dir, device)

    data_path = args.data_path or config.data_path
    samples = load_text_samples(data_path, dedupe=False)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    group_lines = export_group_stream(model, config, samples, device, args.batch_size)
    raw_lines = export_raw_byte_stream(config, samples)

    group_path = output_dir / "group_stream.txt"
    raw_path = output_dir / "raw_byte_stream.txt"
    write_lines(group_path, group_lines)
    write_lines(raw_path, raw_lines)

    summary = {
        "run_dir": str(run_dir),
        "data_path": str(data_path),
        "device": str(device),
        "samples": len(samples),
        "patch_size": config.patch_size,
        "bit_groups": list(config.bit_groups),
        "group_stream_path": str(group_path),
        "raw_byte_stream_path": str(raw_path),
        "group_stream_preview": group_lines[0] if group_lines else "",
        "raw_byte_stream_preview": raw_lines[0] if raw_lines else "",
        "training_best_metrics": payload.get("metrics", {}),
    }
    text = json.dumps(summary, indent=2)
    (output_dir / "export_summary.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
