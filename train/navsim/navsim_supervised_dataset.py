from __future__ import annotations

import os
import json
import gzip
import pickle
from typing import List, Dict, Any, Sequence

import torch
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.utils import rank0_print

# We need the preprocess helpers already defined in the main training script.
# They live in train.llava.train.train (file path train/llava/train/train.py).
# To avoid circular import issues when this module is imported *before* that file
# (rare), we lazily import them inside __getitem__.


def _lazy_import_preprocs():
    """Dynamically import the training helper functions without causing circular imports."""

    try:
        import importlib

        # Import the *module* (not the `train` function) so we can access helpers defined at
        # the top-level of train/llava/train/train.py
        _train_module = importlib.import_module("llava.train.train")  # type: ignore

    except ModuleNotFoundError:
        # The package might not yet be discoverable on PYTHONPATH when this file is imported.
        # Fall back to loading the file directly via its absolute path.
        import importlib.util, sys, pathlib

        root = pathlib.Path(__file__).resolve().parents[2] / "llava" / "train" / "train.py"
        spec = importlib.util.spec_from_file_location("llava.train.train", root)
        if spec is None or spec.loader is None:
            raise

        _train_module = importlib.util.module_from_spec(spec)  # type: ignore
        sys.modules[spec.name] = _train_module  # type: ignore
        spec.loader.exec_module(_train_module)  # type: ignore

    # Finally, expose the two preprocess helpers.
    return _train_module.preprocess_multimodal, _train_module.preprocess


class NavsimSupervisedDataset(torch.utils.data.Dataset):
    """Load NavSim cache samples and convert to LLaDA-V supervised fine-tuning tuples.

    Each element is a dict with *input_ids*, *labels* (LongTensor) and optional *image* placeholders.
    """

    def __init__(
        self,
        *,
        json_path: str,
        cache_root: str,
        tokenizer,
        data_args,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.cache_root = os.path.abspath(cache_root)

        with open(json_path, "r", encoding="utf-8") as f:
            try:
                self.sample_rel = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                self.sample_rel = [ln.strip() for ln in f if ln.strip()]

        rank0_print(f"[NavSim] total {len(self.sample_rel)} samples loaded.")

    # ----------------------------- utils ----------------------------
    @staticmethod
    def _load_pickle(path: str) -> Dict[str, Any]:
        with gzip.open(path, "rb") as gf:
            return pickle.load(gf)

    # ----------------------------------------------------------------
    def __len__(self):
        return len(self.sample_rel)

    def __getitem__(self, idx):
        preprocess_multimodal, preprocess = _lazy_import_preprocs()

        sample_dir = os.path.join(self.cache_root, self.sample_rel[idx])
        feat_pkl = os.path.join(sample_dir, "transfuser_feature.gz")
        tgt_pkl = os.path.join(sample_dir, "transfuser_target.gz")

        feat = self._load_pickle(feat_pkl)
        tgt = self._load_pickle(tgt_pkl)

        meta_images: List[str] = [str(p) for p in feat["camera_feature"].values()]

        # Build prompt strings -------------------------------------------------
        # prompt_system = (
        #     "As an experienced driver, you are operating a vehicle and have access to a "
        #     "real-time continuous view of the left, front and right from your first-person perspective."
        # )
        directions = ["go left", "go straight", "go right", "unknown"]
        driving_onehot = feat["status_feature"][:4]
        velocity = feat["status_feature"][4:6]
        dir_idx = int(torch.argmax(driving_onehot).item())
        dir_text = directions[dir_idx]
        speed = int((velocity[0].item() ** 2 + velocity[1].item() ** 2) ** 0.5 + 0.5)
        prompt_user = (
            "Please predict the most likely future trajectory of your vehicle on the ego coordinate system, "
            "considering factors such as traffic rules, road conditions, and surrounding vehicles. "
            f"The current speed is about {speed} m/s, and you plan to take the following action: {dir_text}."
        )

        image_tokens = "\n".join([DEFAULT_IMAGE_TOKEN] * len(meta_images))
        human_text = f"{image_tokens}\n{prompt_user}"

        # Future trajectory to string
        traj = tgt["trajectory"][:, :2].tolist()
        traj_str = str([(round(x, 2), round(y, 2)) for x, y in traj])

        conversations = [[{"from": "human", "value": human_text}, {"from": "gpt", "value": traj_str}]]

        # multimodal preprocess ----------------------------------------------
        sources = preprocess_multimodal(conversations, self.data_args)
        data_dict = preprocess(sources, self.tokenizer, has_image=True)

        # ------------------------------------------------------------------
        # Image tensors (use LLaDA's image_processor similar to LazySupervisedDataset)
        # ------------------------------------------------------------------
        def _process_image(path: str):
            from PIL import Image as _PIL
            processor = self.data_args.image_processor
            try:
                img = _PIL.open(path).convert("RGB")
            except Exception as exc:
                raise FileNotFoundError(f"Failed to open image {path}: {exc}")

            image_aspect_ratio = getattr(self.data_args, "image_aspect_ratio", "square")
            if image_aspect_ratio == "pad":
                def _expand2square(pil_img, bg):
                    w,h = pil_img.size
                    if w == h:
                        return pil_img
                    size = max(w,h)
                    new_img = _PIL.new(pil_img.mode, (size,size), bg)
                    paste_pos = ((size-w)//2, (size-h)//2)
                    new_img.paste(pil_img, paste_pos)
                    return new_img
                img = _expand2square(img, tuple(int(x*255) for x in processor.image_mean))
                tensor = processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
            else:
                tensor = processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
            return (tensor, img.size, "image")

        image_tuples = [_process_image(p) for p in meta_images]
        data_dict["image"] = image_tuples

        return {
            "input_ids": data_dict["input_ids"][0],
            "labels": data_dict["labels"][0],
            "image": image_tuples,
            "id": idx,
        } 