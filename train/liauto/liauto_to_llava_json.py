import os
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Local imports from existing codebase
from liauto.liauto_dataset import OnlineDataset  # type: ignore
from liauto.utils.config import Config  # Existing LiAuto helper
from llava.bev_utils import coord_to_token


def trajectory_to_bev_tokens(traj: np.ndarray, max_points: int = 8) -> str:
    """Convert an array [[x,y,z?], ...] to space-separated BEV tokens."""
    tokens = []
    for x, y, *_ in traj[:max_points]:
        tokens.append(coord_to_token(float(x), float(y)))
    return " ".join(tokens)


def save_image(arr: np.ndarray, path: Path):
    """Save HWC uint8 RGB numpy image to *path* (PNG)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path, format="PNG")


def main(args):
    # 1. Build LiAuto dataset
    with open(args.dataset_cfg, "r", encoding="utf-8") as f:
        dataset_cfg = json.load(f)
    liauto_ds = OnlineDataset(dataset_cfg, test_mode=True)  # history_frames=0

    out_json = []
    img_root = Path(args.output_imgs)
    img_root.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(len(liauto_ds)), desc="Converting"):
        features, targets = liauto_ds[idx]  # prepare_features + targets
        cams = [features["camera_feature"]]  # Not raw imgs; we need raw
        # We must re-extract raw images; use prepare_raw_data
        raw = liauto_ds.prepare_raw_data(idx)
        imgs = raw["images"]  # dict with keys left/front/right as ndarray
        left, front, right = imgs["left"], imgs["front"], imgs["right"]

        # Save images
        base_name = f"{idx:07d}"
        paths = []
        for cam, array in zip(["left", "front", "right"], [left, front, right]):
            p = img_root / f"{base_name}_{cam}.png"
            save_image(array, p)
            paths.append(str(p.relative_to(img_root.parent)))

        # Build conversation
        human_msg = "<image>\n<image>\n<image>\n请根据当前驾驶环境预测未来轨迹。"
        traj_tokens = trajectory_to_bev_tokens(targets["trajectory"].numpy())
        gpt_msg = traj_tokens

        sample = {
            "id": base_name,
            "image": paths,  # list of 3
            "conversations": [
                {"from": "human", "value": human_msg},
                {"from": "gpt", "value": gpt_msg},
            ],
        }
        out_json.append(sample)

    # 3. Save JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"[✓] Saved {len(out_json)} samples to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LiAuto dataset to LLava/LLaDA json format")
    parser.add_argument("--dataset_cfg", required=True, help="Path to LiAuto dataset config (JSON)")
    parser.add_argument("--output_json", required=True, help="Output json file for training")
    parser.add_argument("--output_imgs", required=True, help="Directory to save extracted images")
    args = parser.parse_args()
    main(args) 