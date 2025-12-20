#!/usr/bin/env python3
"""
Download HPSv2 checkpoints into a dedicated local folder.
"""

import argparse
import pathlib
from huggingface_hub import hf_hub_download


VERSION_TO_FILENAME = {
    "v2.0": "HPS_v2_compressed.pt",
    "v2.1": "HPS_v2.1_compressed.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch HPSv2 weights from Hugging Face."
    )
    parser.add_argument(
        "--version",
        choices=VERSION_TO_FILENAME.keys(),
        default="v2.1",
        help="Model release to download (default: v2.1).",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "weights",
        help="Directory where the weights will be stored (default: RLEPD/weights).",
    )
    parser.add_argument(
        "--symlink-cache",
        action="store_true",
        help="If set, keep Hugging Face cache symlinks instead of copying files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = VERSION_TO_FILENAME[args.version]
    path = hf_hub_download(
        repo_id="xswu/HPSv2",
        filename=filename,
        repo_type="model",
        local_dir=str(args.output_dir),
        local_dir_use_symlinks=args.symlink_cache,
    )
    print(path)


if __name__ == "__main__":
    main()



# python download_hpsv2_weights.py