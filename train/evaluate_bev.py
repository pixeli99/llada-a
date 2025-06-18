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
from llava.bev_utils import decode_bev_tokens  # Convert "<bev_idx>" token pairs to (x, y)

# ---------- Utility functions ----------
_FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"                 # support scientific notation
_TUPLE_RE = re.compile(r"\(\s*(" + _FLOAT + r")\s*,\s*(" + _FLOAT + r")\s*\)")

def parse_xy_list(text: str) -> List[Tuple[float, float]]:
    """
    Parse a list of (x, y) coordinates from the model output.

    1) Prefer the official `decode_bev_tokens` (most reliable when `<bev_idx>` tokens are present).
    2) Fallback to a regex such as "[(1.2, 3.4), (5.6,7.8)]" when no BEV token is found.
    """
    # First try the official conversion (most reliable when `<bev_idx>` tokens are present)
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
        attn_implementation="sdpa", device_map=args.device_map)
    model.eval()

    # Determine primary device (first weight's device works for HF sharded models)
    first_device = next(model.parameters()).device

    # 3. Inference & evaluation
    conv_tmpl = "llava_llada"
    global_ade, global_fde = [], []
    rows = []  # For optional CSV export

    # ----------------------------------------------------------
    # 2) Organise samples into batches for faster inference
    # ----------------------------------------------------------
    B = max(1, args.batch_size)
    num_samples = len(gt_data)

    for i in tqdm(range(0, num_samples, B), desc="Evaluating"):
        batch = gt_data[i : i + B]

        sids, batch_prompts, batch_imgs, batch_gts, batch_sizes = [], [], [], [], []

        # 2.1   Build prompts + load images
        for sample in batch:
            sid   = sample["id"]
            img_p = Path(args.image_root) / sample["image"]

            try:
                img = Image.open(img_p).convert("RGB")
            except FileNotFoundError:
                print(f"[WARN] image {img_p} not found, skip."); continue

            prompt = sample["conversations"][0]["value"]
            gt_traj = sample.get("gt") or sample.get("future_trajectory") \
                      or parse_xy_list(sample["conversations"][-1]["value"])
            if not isinstance(gt_traj[0], tuple):
                gt_traj = parse_xy_list(str(gt_traj))

            # Conversation prompt
            conv = conv_templates[conv_tmpl].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
            conv.append_message(conv.roles[1], None)
            prompt_q = conv.get_prompt()

            sids.append(sid)
            batch_prompts.append(prompt_q)
            batch_imgs.append(img)
            batch_gts.append(gt_traj)
            batch_sizes.append(img.size)

        if len(sids) == 0:
            continue  # all missing images

        # 2.2   Tokenise & pad
        input_id_list = [tokenizer_image_token(p, tokenizer,
                                               IMAGE_TOKEN_INDEX, return_tensors="pt")
                          for p in batch_prompts]

        pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        # Input tensors must be on the same device as the model's first shard.
        input_ids = torch.nn.utils.rnn.pad_sequence(input_id_list, batch_first=True,
                                                    padding_value=pad_token).to(first_device)

        # 2.3   Process images together
        images = process_images(batch_imgs, img_proc, model.config)
        if isinstance(images, list):
            images = [im.to(dtype=torch.float16, device=first_device) for im in images]
        else:
            images = images.to(dtype=torch.float16, device=first_device)

        # 2.4   Run model
        start = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                images=images,
                image_sizes=batch_sizes,
                steps=128, gen_length=128, block_length=128,
                tokenizer=tokenizer, stopping_criteria=['<|eot_id|>']
            )
        infer_time = (time.time() - start) / len(sids)  # per-sample time

        # 2.5   Decode + metrics
        pred_texts = tokenizer.batch_decode(out_ids, skip_special_tokens=False)

        for sid, pred_text, gt_traj in zip(sids, pred_texts, batch_gts):
            pred_traj = parse_xy_list(pred_text)
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
    parser = argparse.ArgumentParser(description="Evaluate ADE/FDE for LLaDA-V (batched + multi-GPU)")
    parser.add_argument("--gt_json",   required=True,  help="Path to GT json")
    parser.add_argument("--image_root",default=".",    help="Root dir of images")
    parser.add_argument("--pretrained",default="GSAI-ML/LLaDA-V",
                        help="HF repo or local ckpt dir")
    parser.add_argument("--device_map",default="auto",
                        help="Model device map, e.g. 'auto' or 'cuda:0,1' (passed to HF load_pretrained_model)")
    parser.add_argument("--batch_size",type=int, default=4,
                        help="Number of samples per generation batch")
    parser.add_argument("--csv_out",   default=None,
                        help="Save per-sample results as CSV")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    main(args)