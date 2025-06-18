#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_bev.py — Evaluate ADE/FDE for LLaDA-V trajectory predictions.

Example:
  python evaluate_bev.py \
         --gt_json data/gt.json \
         --image_root data/ \
         --pretrained GSAI-ML/LLaDA-V \
         --device cuda:0 \
         --csv_out results.csv
"""
import argparse, json, re, time, csv, os, sys, warnings
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.bev_utils import decode_bev_tokens  # Convert "<bev_x_y>" tokens to (x, y)

# ---------- Utility functions ----------
_FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"                 # support scientific notation
_TUPLE_RE = re.compile(r"\(\s*(" + _FLOAT + r")\s*,\s*(" + _FLOAT + r")\s*\)")

def parse_xy_list(text: str) -> List[Tuple[float, float]]:
    """
    Parse a list of (x, y) coordinates from the model output.

    1) Prefer the official `decode_bev_tokens` (most reliable when `<bev_x_y>` tokens are present).
    2) Fallback to a regex such as "[(1.2, 3.4), (5.6,7.8)]" when no BEV token is found.
    """
    # First try the official conversion (most reliable when `<bev_x_y>` tokens are present)
    decoded = decode_bev_tokens(text)
    if decoded and isinstance(decoded, list) and isinstance(decoded[0], tuple):
        return decoded

    # Then fall back to regex such as "[(1.2, 3.4), (5.6,7.8)]"
    matches = _TUPLE_RE.findall(text)
    return [(float(x), float(y)) for x, y in matches]

def ade_fde(pred: List[Tuple[float, float]],
            gt:   List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Compute ADE and FDE for a single sequence.

    If the lengths differ, only the first `min(len(pred), len(gt))` points are compared.
    """
    n = min(len(pred), len(gt))
    if n == 0:
        return float("nan"), float("nan")
    diffs = [(pred[i][0] - gt[i][0], pred[i][1] - gt[i][1]) for i in range(n)]
    dists = [(dx**2 + dy**2) ** 0.5 for dx, dy in diffs]
    ade = sum(dists) / n
    fde = dists[-1]
    return ade, fde

# ---------- Main pipeline ----------
def main(args):
    # 1. Load GT JSON
    with open(args.gt_json, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # 2. Load model
    tokenizer, model, img_proc, _ = load_pretrained_model(
        args.pretrained, None, "llava_llada",
        attn_implementation="sdpa", device_map=args.device)
    model.eval()

    # 3. Inference & evaluation
    conv_tmpl = "llava_llada"
    global_ade, global_fde = [], []
    rows = []  # For optional CSV export

    for sample in tqdm(gt_data, desc="Evaluating"):
        sid   = sample["id"]
        img_p = Path(args.image_root) / sample["image"]
        prompt = sample["conversations"][0]["value"]  # user prompt
        gt_traj = sample.get("gt") or sample.get("future_trajectory") \
                  or parse_xy_list(sample["conversations"][-1]["value"])
        # Replace this line if GT is stored in another field

        if not isinstance(gt_traj[0], tuple):
            # GT may still be a string — parse it
            gt_traj = parse_xy_list(str(gt_traj))

        # ---------- Inference ----------
        try:
            img = Image.open(img_p).convert("RGB")
        except FileNotFoundError:
            print(f"[WARN] image {img_p} not found, skip."); continue

        # 3.1 Build conversation prompt
        conv = conv_templates[conv_tmpl].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        prompt_q = conv.get_prompt()

        # 3.2 Prepare inputs
        input_ids = tokenizer_image_token(prompt_q, tokenizer,
                                          IMAGE_TOKEN_INDEX, return_tensors="pt"
                                          ).unsqueeze(0).to(args.device)
        images = process_images([img], img_proc, model.config)
        images = [_i.to(dtype=torch.float16, device=args.device)
                  for _i in images]
        img_sizes = [img.size]

        # 3.3 Run model
        start = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                images=images,
                image_sizes=img_sizes,
                steps=128, gen_length=128, block_length=128,
                tokenizer=tokenizer, stopping_criteria=['<|eot_id|>'])
        infer_time = time.time() - start

        # 3.4 Decode coordinates
        pred_text = tokenizer.batch_decode(out_ids, skip_special_tokens=False)[0]
        pred_traj = parse_xy_list(pred_text)

        # 4. Error metrics
        ade, fde = ade_fde(pred_traj, gt_traj)
        global_ade.append(ade); global_fde.append(fde)

        rows.append({
            "id": sid,
            "ade": ade,
            "fde": fde,
            "infer_time_sec": infer_time,
            "pred": pred_traj,
            "gt": gt_traj
        })

    mean_ade = sum(global_ade) / len(global_ade)
    mean_fde = sum(global_fde) / len(global_fde)
    print(f"\n========  Overall Results  ========\n"
          f"Samples evaluated : {len(global_ade)}\n"
          f"Mean ADE          : {mean_ade:.4f}\n"
          f"Mean FDE          : {mean_fde:.4f}")

    # 5. Optional CSV export
    if args.csv_out:
        keys = ["id", "ade", "fde", "infer_time_sec", "pred", "gt"]
        with open(args.csv_out, "w", newline='', encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        print(f"[✓] Per‑sample details saved to {args.csv_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ADE/FDE for LLaDA‑V")
    parser.add_argument("--gt_json",   required=True,  help="Path to GT json")
    parser.add_argument("--image_root",default=".",    help="Root dir of images")
    parser.add_argument("--pretrained",default="GSAI-ML/LLaDA-V",
                        help="HF repo or local ckpt dir")
    parser.add_argument("--device",    default="cuda:0",
                        help="cuda:0 | cuda:1 | cpu")
    parser.add_argument("--csv_out",   default=None,
                        help="Save per‑sample results as CSV")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    main(args)