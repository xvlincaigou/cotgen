#!/usr/bin/env python
"""
Generate images from COT metadata (geneval/cot_to_sana.py) using SanaPipeline.

Output layout (per metadata line):
<output_dir>/
    00000/
        metadata.jsonl  # original metadata line
        grid.png        # optional
        samples/
            0000.png
            0001.png
            0002.png
            0003.png

Example:
    python scripts/generate_from_cot.py \
        --cot-file ../geneval/prompts/cot_outputs/cot_metadata.jsonl \
        --output-dir outputs/cot_images \
        --diffusers-id /root/autodl-tmp/huggingface_cache/hub/models--Efficient-Large-Model--Sana_600M_1024px_diffusers/snapshots/b3ddc862873249650ed9f6ba90e92ede3083bb0d \
        --height 1024 --width 1024 --guidance-scale 4.5 --steps 50 --num-images 4 --save-grid
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from diffusers import SanaPipeline
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from einops import rearrange
from PIL import Image


def load_metadata(path: Path) -> List[Tuple[str, dict, str]]:
    """Return list of (prompt, obj, raw_line)."""
    items: List[Tuple[str, dict, str]] = []
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot-file", type=str, required=True, help="Path to cot_metadata.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/cot_images", help="Where to save images")
    parser.add_argument("--diffusers-id", type=str, default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num-images", type=int, default=4, help="Images per prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-grid", action="store_true", help="Also save grid.png per prompt")
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

    for idx, (prompt, meta, raw_line) in enumerate(entries):
        seed = args.seed + idx
        out = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            num_images_per_prompt=args.num_images,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        images = getattr(out, "images", out)

        prompt_dir = out_dir / f"{idx:05d}"
        samples_dir = prompt_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata line
        with (prompt_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
            f.write(raw_line + "\n")

        stacked = []
        for j, pil_img in enumerate(images):
            name = f"{j:04d}.png"
            pil_img.save(samples_dir / name)
            stacked.append(ToTensor()(pil_img))

        if args.save_grid and stacked:
            grid = make_grid(torch.stack(stacked, dim=0), nrow=args.num_images)
            grid = (grid * 255.0).clamp(0, 255).byte()
            grid_img = rearrange(grid.cpu().numpy(), "c h w -> h w c")
            Image.fromarray(grid_img).save(prompt_dir / "grid.png")

    print(f"Saved images to {out_dir}")


if __name__ == "__main__":
    main()
