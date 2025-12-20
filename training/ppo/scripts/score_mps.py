"""
Utility CLI to compute MPS (Multi-dimensional Preference Score) for生成图.

用法示例：
    python -m training.ppo.scripts.score_mps \\
        --images exps/<run-id>/samples \\
        --prompts src/prompts/test.txt \\
        --weights weights/MPS_overall_checkpoint.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable, List, Sequence
import inspect
import types

import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor

warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm.*")


def _load_prompts(path: Path) -> List[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            prompts = [row.get("text", "").strip() for row in reader if row.get("text")]
    else:
        with path.open("r", encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise RuntimeError(f"未能从 {path} 读取到任何 prompt。")
    return prompts


def _collect_images(directory: Path, pattern: str) -> List[Path]:
    files = sorted(directory.glob(pattern))
    if not files:
        raise RuntimeError(f"在 {directory} 下未找到匹配 {pattern} 的图像文件。")
    return files


def _default_mps_root() -> Path:
    return Path(__file__).resolve().parents[3] / "MPS"


class _MPSScorer:
    def __init__(
        self,
        weights: Path,
        device: torch.device,
        processor_name: str,
        condition: str,
        mps_root: Path,
        mask_threshold: float,
        enable_fp16: bool,
    ) -> None:
        self.device = device
        self.mask_threshold = mask_threshold
        self.enable_fp16 = enable_fp16 and device.type == "cuda"
        self.processor = CLIPImageProcessor.from_pretrained(processor_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            processor_name, trust_remote_code=True
        )

        if not mps_root.exists():
            raise RuntimeError(f"MPS 源码目录 {mps_root} 不存在。")
        if str(mps_root) not in sys.path:
            sys.path.insert(0, str(mps_root))

        try:
            self.model = torch.load(weights, map_location="cpu", weights_only=False)
        except TypeError:
            self.model = torch.load(weights, map_location="cpu")
        self.model.eval().to(self.device)

        clip_config = getattr(getattr(self.model, "model", None), "config", None)
        if clip_config is not None:
            self._ensure_config_defaults(
                clip_config,
                [
                    ("_output_attentions", False),
                    ("_output_hidden_states", False),
                    ("_use_return_dict", True),
                    ("_attn_implementation", "eager"),
                    ("_attn_implementation_internal", "eager"),
                ],
            )
            for sub_name in ("text_config", "vision_config"):
                sub_cfg = getattr(clip_config, sub_name, None)
                if sub_cfg is not None:
                    self._ensure_config_defaults(
                        sub_cfg,
                        [
                            ("_attn_implementation", "eager"),
                            ("_attn_implementation_internal", "eager"),
                        ],
                    )

        self.condition_tokens = self._tokenize(condition).to(self.device)

        clip_backbone = getattr(self.model, "model", None)
        if clip_backbone is not None:
            text_model = getattr(clip_backbone, "text_model", None)
            vision_model = getattr(clip_backbone, "vision_model", None)
            for module in (text_model, vision_model):
                self._ensure_return_dict_kwarg(module)
                self._ensure_attention_compat(module)
            self._ensure_text_defaults(text_model)

    def _ensure_config_defaults(self, config, entries):
        for field, default in entries:
            if not hasattr(config, field):
                setattr(config, field, default)

    def _ensure_return_dict_kwarg(self, module: torch.nn.Module | None) -> None:
        if module is None:
            return
        try:
            sig = inspect.signature(module.forward)
        except (ValueError, TypeError):
            return
        if "return_dict" in sig.parameters:
            return

        orig_forward = module.forward

        def wrapped_forward(self, *args, **kwargs):
            kwargs.pop("return_dict", None)
            return orig_forward(*args, **kwargs)

        module.forward = types.MethodType(wrapped_forward, module)

    def _ensure_attention_compat(self, module: torch.nn.Module | None) -> None:
        if module is None:
            return
        for child in module.modules():
            if child.__class__.__name__ == "CLIPAttention":
                if not hasattr(child, "is_causal"):
                    child.is_causal = False

    def _ensure_text_defaults(self, module: torch.nn.Module | None) -> None:
        if module is None:
            return
        defaults = {
            "eos_token_id": 2,
            "bos_token_id": 0,
            "pad_token_id": 0,
        }
        for name, value in defaults.items():
            if not hasattr(module, name):
                setattr(module, name, value)

    def _tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

    def _process_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        tensor = self.processor(image, return_tensors="pt")["pixel_values"]
        if self.enable_fp16:
            tensor = tensor.half()
        return tensor.to(self.device)

    @torch.no_grad()
    def score(self, prompt: str, image_path: Path) -> float:
        text_ids = self._tokenize(prompt).to(self.device)
        condition_ids = self.condition_tokens
        if condition_ids.shape[0] != text_ids.shape[0]:
            condition_ids = condition_ids.repeat(text_ids.shape[0], 1)
        image_tensor = self._process_image(image_path)

        text_f, text_feat = self.model.model.get_text_features(text_ids)
        image_f = self.model.model.get_image_features(image_tensor)
        condition_f, _ = self.model.model.get_text_features(condition_ids)

        sim_text_condition = torch.einsum("b i d, b j d -> b j i", text_f, condition_f)
        sim_text_condition = torch.max(sim_text_condition, dim=1, keepdim=True)[0]
        sim_text_condition = sim_text_condition / sim_text_condition.max()
        mask = torch.where(
            sim_text_condition > self.mask_threshold, 0.0, float("-inf")
        ).repeat(1, image_f.shape[1], 1)
        mask = mask.to(image_f.dtype)

        image_features = self.model.cross_model(image_f, text_f, mask)[:, 0, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        score = self.model.logit_scale.exp() * (text_feat @ image_features.T)
        return score[0].item()


def _summarize(scores: torch.Tensor) -> dict:
    stats = {
        "count": int(scores.shape[0]),
        "mean": float(scores.mean().item()),
        "std": float(scores.std(unbiased=False).item()) if scores.numel() > 1 else 0.0,
        "min": float(scores.min().item()),
        "max": float(scores.max().item()),
    }
    return stats


def _evaluate(
    scorer: _MPSScorer,
    image_files: Sequence[Path],
    prompts: Sequence[str],
    batch_size: int,
) -> tuple[torch.Tensor, dict]:
    if len(image_files) != len(prompts):
        raise RuntimeError(f"图像数量 ({len(image_files)}) 与 prompt 数量 ({len(prompts)}) 不一致。")

    values: List[float] = []
    start = time.time()

    for offset in range(0, len(image_files), batch_size):
        chunk_files = image_files[offset : offset + batch_size]
        chunk_prompts = prompts[offset : offset + len(chunk_files)]
        for path, prompt in zip(chunk_files, chunk_prompts):
            values.append(float(scorer.score(prompt, path)))

    duration = time.time() - start
    return torch.tensor(values, dtype=torch.float32), {
        "duration": duration,
        "num_images": len(values),
        "batch_size": batch_size,
        "device": str(scorer.device),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute MPS scores for generated images.")
    parser.add_argument("--images", type=Path, required=True, help="包含生成图像的目录。")
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="匹配图像文件的 glob pattern（默认: *.png）。",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="Prompt 列表文件（CSV 需包含 text 列，或纯文本每行一个 prompt）。",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/MPS_overall_checkpoint.pth"),
        help="MPS checkpoint 路径。",
    )
    parser.add_argument(
        "--mps-root",
        type=Path,
        default=_default_mps_root(),
        help="MPS 源码根目录（用于导入 trainer 模块）。",
    )
    parser.add_argument(
        "--processor-name",
        type=str,
        default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        help="CLIP tokenizer/image processor 名称。",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=(
            "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, "
            "hands, limbs, structure, instance, texture, quantity, attributes, position, number, "
            "location, word, things."
        ),
        help="用于 overall 评测的 condition 描述。",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.3,
        help="构建 cross attention 掩码时的阈值（默认 0.3）。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="顺序评估时的批大小，用于拆分列表。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行 MPS 评估的设备。",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="禁用 FP16 推理（默认在 CUDA 上启用 FP16）。",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="可选：将结果写入 JSON 文件。",
    )

    args = parser.parse_args(argv)

    images_dir = args.images.resolve()
    prompts_path = args.prompts.resolve()
    weights_path = args.weights.expanduser()
    if not weights_path.exists():
        raise RuntimeError(f"未找到 MPS 权重文件：{weights_path}")

    image_files = _collect_images(images_dir, args.pattern)
    prompts = _load_prompts(prompts_path)

    device = torch.device(args.device)
    scorer = _MPSScorer(
        weights=weights_path,
        device=device,
        processor_name=args.processor_name,
        condition=args.condition,
        mps_root=args.mps_root.expanduser(),
        mask_threshold=args.mask_threshold,
        enable_fp16=not args.disable_amp,
    )

    scores, metadata = _evaluate(scorer, image_files, prompts, args.batch_size)
    stats = _summarize(scores)

    result = {
        "images_dir": str(images_dir),
        "pattern": args.pattern,
        "prompts_file": str(prompts_path),
        "weights": str(weights_path),
        "mps_root": str(args.mps_root.expanduser()),
        "stats": stats,
        "metadata": metadata,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":  # pragma: no cover
    main()
