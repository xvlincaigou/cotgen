#!/usr/bin/env python
"""
Generate one image per label from COT metadata, using SanaPipeline.

Behavior:
- Read a JSONL (cot_metadata.jsonl) where each line has at least `prompt` or `final_prompt`, and optional `box_2d` with `label`.
- For每个metadata条目，对其中的每个标签生成一张图（不做拼接）。
- 输出结构：
  <output_dir>/<entry_idx>/
    metadata.jsonl  # 原始行
    <label>.png     # 每个标签一张图

Example:
    python scripts/generate_from_cot_labels.py \
        --cot-file ../geneval/prompts/cot_outputs/cot_metadata.jsonl \
        --output-dir outputs/cot_images_labels \
        --diffusers-id /root/autodl-tmp/huggingface_cache/hub/models--Efficient-Large-Model--Sana_600M_1024px_diffusers/snapshots/b3ddc862873249650ed9f6ba90e92ede3083bb0d \
        --height 1024 --width 1024 --guidance-scale 4.5 --steps 50 --seed 42
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from diffusers import SanaPipeline
from PIL import Image


def load_metadata(path: Path) -> List[Tuple[str, Dict, str]]:
    """Return list of (prompt_text, obj, raw_line)."""
    items: List[Tuple[str, Dict, str]] = []
    for line in path.read_text().splitlines():
        raw = line.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        prompt = obj.get("final_prompt") or obj.get("prompt")
        if not prompt:
            continue
        items.append((prompt, obj, raw))
    return items


def extract_labels(meta: Dict) -> List[str]:
    labels: List[str] = []
    for box in meta.get("box_2d", []) or []:
        if isinstance(box, dict):
            label = box.get("label") or box.get("class")
            if label:
                labels.append(str(label))
    # 去重并保持顺序
    seen = set()
    uniq = []
    for lb in labels:
        if lb not in seen:
            seen.add(lb)
            uniq.append(lb)
    if not uniq:
        uniq = ["sample"]
    return uniq


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_") or "sample"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot-file", type=str, required=True, help="Path to cot_metadata.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/cot_images_labels", help="Where to save images")
    parser.add_argument("--diffusers-id", type=str, default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cot_path = Path(args.cot_file).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_metadata(cot_path)
    if not entries:
        print("No prompts found in COT file.")
        return

    pipe = SanaPipeline.from_pretrained(
        args.diffusers_id,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)

    for idx, (base_prompt, meta, raw_line) in enumerate(entries):
        labels = extract_labels(meta)
        prompt_dir = out_dir / f"{idx:05d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        # 保存原始行
        with (prompt_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
            f.write(raw_line + "\n")

        for j, label in enumerate(labels):
            prompt = f"{base_prompt}. Focus on {label}."
            seed = args.seed + idx * 1000 + j
            out = pipe(
                prompt=prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
                num_images_per_prompt=1,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            )
            image = getattr(out, "images", out)[0]
            name = slugify(label) + ".png"
            image.save(prompt_dir / name)

    print(f"Saved images to {out_dir}")


if __name__ == "__main__":
    main()
