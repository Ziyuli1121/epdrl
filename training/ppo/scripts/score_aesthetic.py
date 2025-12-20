"""
Utility CLI to compute aesthetic scores for a directory of generated images.

Usage example:
    python -m training.ppo.scripts.score_aesthetic \\
        --images exps/<run-id>/samples \\
        --prompts prompts.csv \\
        --weights weights/sac+logos+ava1-l14-linearMSE.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import torch
from PIL import Image

from training.ppo.reward_models.aesthetic_predictor_v2 import AestheticV2Model, torch_normalized

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


def _summarize(scores: torch.Tensor) -> dict:
    stats = {
        "count": int(scores.shape[0]),
        "mean": float(scores.mean().item()),
        "std": float(scores.std(unbiased=False).item()) if scores.numel() > 1 else 0.0,
        "min": float(scores.min().item()),
        "max": float(scores.max().item()),
    }
    return stats


@dataclass
class AestheticReward:
    clip_encoder: torch.nn.Module
    predictor: torch.nn.Module
    preprocess: Callable[[Image.Image], torch.Tensor]
    normalize_fn: Callable[[torch.Tensor], torch.Tensor]
    device: torch.device

    def score_paths(self, paths: Sequence[Path]) -> torch.Tensor:
        images: List[Image.Image] = []
        for path in paths:
            image = Image.open(path)
            images.append(image.convert("RGB"))

        try:
            batch = torch.stack([self.preprocess(img) for img in images], dim=0).to(self.device)
            with torch.no_grad():
                embeds = self.clip_encoder.encode_image(batch)
                embeds = self.normalize_fn(embeds).to(torch.float32)
                scores = self.predictor(embeds).squeeze(-1)
        finally:
            for image in images:
                image.close()

        return scores.detach().cpu()


def _build_reward(args: argparse.Namespace) -> tuple[AestheticReward, str]:
    cache_dir = args.cache_dir.expanduser() if args.cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("CLIP_HOME", str(cache_dir))

    weights_path = args.weights.expanduser()
    if not weights_path.exists():
        raise RuntimeError(f"Aesthetic 权重文件不存在：{weights_path}")

    clip_override = os.environ.get("AESTHETIC_CLIP_PATH")
    base = AestheticV2Model(
        clip_path=clip_override if clip_override else None,
        predictor_path=str(weights_path),
    )

    device = torch.device(args.device)
    clip_encoder = base.clip_encoder.to(device)
    predictor = base.aesthetic_predictor.to(device)
    clip_encoder.eval()
    predictor.eval()

    reward = AestheticReward(
        clip_encoder=clip_encoder,
        predictor=predictor,
        preprocess=base.preprocessor,
        normalize_fn=torch_normalized,
        device=device,
    )
    return reward, str(weights_path)


def _score(
    reward: AestheticReward,
    image_files: Sequence[Path],
    prompts: Sequence[str],
    batch_size: int,
) -> tuple[torch.Tensor, dict]:
    if len(image_files) != len(prompts):
        raise RuntimeError(f"图像文件数量 ({len(image_files)}) 与 prompt 数量 ({len(prompts)}) 不一致。")

    values: List[torch.Tensor] = []
    start = time.time()

    for offset in range(0, len(image_files), batch_size):
        chunk = image_files[offset : offset + batch_size]
        values.append(reward.score_paths(chunk))

    tensor = torch.cat(values, dim=0) if values else torch.empty(0, dtype=torch.float32)
    duration = time.time() - start

    metadata = {
        "duration": duration,
        "num_images": len(image_files),
        "batch_size": batch_size,
        "device": str(reward.device),
    }
    return tensor.to(dtype=torch.float32), metadata


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute aesthetic scores for generated images.")
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
        default=Path("weights/sac+logos+ava1-l14-linearMSE.pth"),
        help="Aesthetic 线性头权重路径。",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("weights"),
        help="CLIP/Aesthetic 模型缓存目录。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="评估时的 batch size。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行 Aesthetic 评估的设备。",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="保留参数以兼容 score_hps 接口；Aesthetic 模型不支持 AMP 切换。",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="可选：将结果写入 JSON 文件。",
    )

    args = parser.parse_args(argv)
    images_dir = args.images.resolve()
    prompts_path = args.prompts.resolve()

    image_files = _collect_images(images_dir, args.pattern)
    prompts = _load_prompts(prompts_path)

    reward, weights_ref = _build_reward(args)
    scores, meta = _score(reward, image_files, prompts, args.batch_size)
    stats = _summarize(scores)

    meta["weights"] = weights_ref

    result = {
        "images_dir": str(images_dir),
        "pattern": args.pattern,
        "prompts_file": str(prompts_path),
        "stats": stats,
        "metadata": meta,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":  # pragma: no cover
    main()
