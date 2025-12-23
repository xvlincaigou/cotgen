#!/usr/bin/env python
"""
Generate images from prompts using diffusers SanaPipeline, saving in geneval format.

Output layout (per prompt):
<output_dir>/
    00000/
        metadata.jsonl  # the matching metadata line (or prompt-only JSON)
        grid.png        # optional
        samples/
            0000.png
            0001.png
            0002.png
            0003.png

Examples:
    # Use plain prompts.txt (one prompt per line)
    python scripts/generate_from_prompts.py \
        --prompt-file ../geneval/prompts/generation_prompts.txt \
        --output-dir outputs/vanilla_images \
        --diffusers-id /root/autodl-tmp/huggingface_cache/hub/models--Efficient-Large-Model--Sana_600M_1024px_diffusers/snapshots/b3ddc862873249650ed9f6ba90e92ede3083bb0d

    # Use evaluation_metadata.jsonl (geneval format) to keep metadata lines
    python scripts/generate_from_prompts.py \
        --metadata-file ../geneval/prompts/evaluation_metadata.jsonl \
        --output-dir outputs/cot_images \
        --num-images 4 --save-grid
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


def load_prompts(path: Path) -> List[str]:
    prompts: List[str] = []
    for line in path.read_text().splitlines():
        prompt = line.strip()
        if prompt:
            prompts.append(prompt)
    return prompts


def load_metadata(path: Path) -> List[Tuple[dict, str]]:
    """Load metadata jsonl, keeping both dict and raw line."""
    items: List[Tuple[dict, str]] = []
    for line in path.read_text().splitlines():
        raw = line.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        items.append((obj, raw))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", type=str, help="Plain text prompts, one per line")
    parser.add_argument("--metadata-file", type=str, help="evaluation_metadata.jsonl; will use its lines and prompts")
    parser.add_argument("--output-dir", type=str, default="outputs/generation_prompts", help="Where to save images")
    parser.add_argument("--diffusers-id", type=str, default="Efficient-Large-Model/Sana_600M_1024px_diffusers")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num-images", type=int, default=4, help="Images per prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-grid", action="store_true", help="Also save grid.png per prompt")
    args = parser.parse_args()

    if not args.prompt_file and not args.metadata_file:
        raise SystemExit("Please provide --prompt-file or --metadata-file")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Tuple[str, dict, str]] = []  # (prompt, metadata_obj, raw_line)

    if args.metadata_file:
        meta_path = Path(args.metadata_file).resolve()
        metas = load_metadata(meta_path)
        for obj, raw in metas:
            prompt = obj.get("final_prompt") or obj.get("prompt")
            if not prompt:
                continue
            entries.append((prompt, obj, raw))
    else:
        prompt_path = Path(args.prompt_file).resolve()
        for prompt in load_prompts(prompt_path):
            meta = {"prompt": prompt}
            raw = json.dumps(meta, ensure_ascii=False)
            entries.append((prompt, meta, raw))

    if not entries:
        print("No prompts found.")
        return

    pipe = SanaPipeline.from_pretrained(
        args.diffusers_id,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)

    for idx, (prompt, meta, raw_line) in enumerate(entries):
        seed = args.seed + idx  # deterministic but different per prompt
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

        # Save metadata jsonl (single line)
        with (prompt_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
            f.write(raw_line + "\n")

        # Save images
        stacked = []
        for j, pil_img in enumerate(images):
            name = f"{j:04d}.png"
            pil_img.save(samples_dir / name)
            stacked.append(ToTensor()(pil_img))

        # Optional grid
        if args.save_grid and stacked:
            grid = make_grid(torch.stack(stacked, dim=0), nrow=args.num_images)
            grid = (grid * 255.0).clamp(0, 255).byte()
            grid_img = rearrange(grid.cpu().numpy(), "c h w -> h w c")
            Image.fromarray(grid_img).save(prompt_dir / "grid.png")

    print(f"Saved images to {out_dir}")


if __name__ == "__main__":
    main()
