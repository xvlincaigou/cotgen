#!/usr/bin/env python
"""
Convert Geneval prompts to chain-of-thought (COT) using a Qwen vLLM server.

Examples:
    python geneval/cot_to_sana.py \
        --input prompts/evaluation_metadata.jsonl \
        --output-dir prompts/cot_outputs \
        --qwen-base-url http://0.0.0.0:8000/v1 \
        --qwen-model qwen25vl-7b
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

DEFAULT_SYSTEM_PROMPT = (
    "You convert short image prompts into structured chain-of-thought. "
    "Return strict JSON with keys 'cot' (list of concise reasoning steps), "
    "'box_2d' (list of objects with normalized [x1,y1,x2,y2] in 0-1 and a label), "
    "and 'final_prompt' (single refined prompt ready for image generation). "
    "Before giving the final prompt, reason about composition and relative size. If a prompt implies size or body-type contrast (e.g., a fat person and a thin person), state it clearly in the COT and reflect it in 'final_prompt'. "
    "If layout is unclear, propose a simple, non-overlapping layout. Limit final_prompt to under 60 words and stay faithful to the user prompt."
)


def load_inputs(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return [{"prompt": ln} for ln in lines]


def build_payload(prompt: str, model: str, system_prompt: str) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }


def parse_completion(data: Dict[str, Any]) -> Tuple[List[str], str, List[Dict[str, Any]]]:
    text = data["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {"cot": [text], "final_prompt": text, "box_2d": []}
    cot = parsed.get("cot") or [parsed.get("final_prompt", text)]
    if isinstance(cot, str):
        cot = [cot]
    final_prompt = parsed.get("final_prompt", cot[-1])
    boxes = parsed.get("box_2d") or []
    if not isinstance(boxes, list):
        boxes = []
    return cot, final_prompt, boxes


def fetch_cot(prompt: str, base_url: str, model: str, api_key: str, system_prompt: str, timeout: int = 120) -> Dict[str, Any]:
    payload = build_payload(prompt, model, system_prompt)
    headers = {"Authorization": f"Bearer {api_key}"}
    url = base_url.rstrip("/") + "/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    cot, final_prompt, boxes = parse_completion(resp.json())
    return {"cot": cot, "final_prompt": final_prompt, "box_2d": boxes}


def merge_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    merged = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        merged[obj["prompt"]] = obj
    return merged


def save_jsonl(objs: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for obj in objs:
            fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="prompts/evaluation_metadata.jsonl", help="Geneval prompts file")
    parser.add_argument("--output-dir", type=str, default="prompts/cot_outputs", help="Where to write COT")
    parser.add_argument("--qwen-base-url", type=str, default="http://0.0.0.0:8000/v1", help="vLLM OpenAI base url")
    parser.add_argument("--qwen-model", type=str, default="qwen25vl-7b", help="Model name used by vLLM")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System prompt for COT")
    parser.add_argument("--api-key", type=str, default=os.getenv("QWEN_API_KEY", "EMPTY"))
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing cot file and re-run")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    cot_path = output_dir / "cot_metadata.jsonl"

    records = load_inputs(input_path)
    existing = merge_existing(cot_path) if not args.overwrite else {}

    combined = existing.copy()
    start = time.time()
    for rec in records:
        prompt = rec.get("prompt")
        if not prompt:
            continue
        if prompt in combined:
            continue
        cot_pack = fetch_cot(
            prompt=prompt,
            base_url=args.qwen_base_url,
            model=args.qwen_model,
            api_key=args.api_key,
            system_prompt=args.system_prompt,
        )
        combined[prompt] = {
            **rec,
            **cot_pack,
            "qwen_model": args.qwen_model,
        }
        if len(combined) % 10 == 0:
            elapsed = time.time() - start
            print(f"Finished {len(combined)} prompts in {elapsed:.1f}s")

    save_jsonl(list(combined.values()), cot_path)
    print(f"Saved COT to {cot_path}")


if __name__ == "__main__":
    main()
