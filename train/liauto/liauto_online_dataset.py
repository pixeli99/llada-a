import json
from pathlib import Path
from typing import Dict, Sequence, Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from liauto.liauto_dataset import OnlineDataset  # type: ignore
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.bev_utils import coord_to_token
from llava.train.train import preprocess_multimodal, preprocess  # reuse helpers
from llava import conversation as conversation_lib


class LiAutoOnlineSupervisedDataset(Dataset):
    """Stream LiAuto data and convert to LLava training samples on-the-fly."""

    def __init__(
        self,
        cfg_path: str,
        tokenizer,
        data_args,
        history_frames: int = 0,
    ):
        super().__init__()
        with open(cfg_path, "r", encoding="utf-8") as f:
            dataset_cfg = json.load(f)
        self.liauto_ds = OnlineDataset(dataset_cfg, history_frames=history_frames, test_mode=True)
        self.tokenizer = tokenizer
        self.data_args = data_args  # Needs image_processor etc.

    # ---------------------------------------------------------------------
    # image helpers
    # ---------------------------------------------------------------------
    def _np_to_pil(self, arr: np.ndarray) -> Image.Image:
        return Image.fromarray(arr.astype(np.uint8))

    def _process_single_image(self, img: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        proc = self.data_args.image_processor
        image_aspect_ratio = self.data_args.image_aspect_ratio
        size = img.size
        if image_aspect_ratio == "highres":
            tensor = process_highres_image(img, proc, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            tensor = process_anyres_image(img, proc, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            tensor = process_highres_image_crop_split(img, self.data_args)
        elif image_aspect_ratio == "pad":
            # square pad
            bg = tuple(int(x * 255) for x in proc.image_mean)
            side = max(img.size)
            padded = Image.new("RGB", (side, side), bg)
            padded.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
            tensor = proc.preprocess(padded, return_tensors="pt")["pixel_values"][0]
        else:
            tensor = proc.preprocess(img, return_tensors="pt")["pixel_values"][0]
        return tensor, size

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.liauto_ds)

    def __getitem__(self, idx):
        # Raw scene dict
        raw = self.liauto_ds.prepare_raw_data(idx)
        # images
        imgs = raw["images"]  # dict left/front/right
        left, front, right = imgs["left"], imgs["front"], imgs["right"]
        pil_left = self._np_to_pil(left)
        pil_front = self._np_to_pil(front)
        pil_right = self._np_to_pil(right)

        img_tensors = []
        img_sizes = []
        for pil in [pil_left, pil_front, pil_right]:
            tensor, size = self._process_single_image(pil)
            img_tensors.append((tensor, size, "image"))
            img_sizes.append(size)

        # Build conversation (single-round)
        human_msg = DEFAULT_IMAGE_TOKEN + "\n" + DEFAULT_IMAGE_TOKEN + "\n" + DEFAULT_IMAGE_TOKEN + "\n请根据当前驾驶环境预测未来轨迹。"
        # targets – future trajectory 8 points sampled every 5 step (already done in prepare_targets in original code)
        traj = raw["future_trajectory"][::5][:8]  # (8,3) array
        traj_tokens = []
        for x, y, _ in traj:
            traj_tokens.append(coord_to_token(float(x), float(y)))
        gpt_msg = " ".join(traj_tokens)

        conv = [
            {"from": "human", "value": human_msg},
            {"from": "gpt", "value": gpt_msg},
        ]

        sources = preprocess_multimodal([conv], self.data_args)
        data_dict = preprocess(sources, self.tokenizer, has_image=True)

        sample = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
            image=img_tensors,
            image_sizes=img_sizes,
            id=raw.get("log_name", idx),
            is_llada=True,
            is_plain=False,
        )
        return sample 