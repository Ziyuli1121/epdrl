#!/usr/bin/env python3
"""
Check Stable Diffusion 1.x (CLIP ViT-L/14) tokenizer lengths for prompt files.
This script mirrors the Hugging Face `CLIPTokenizer` configuration shipped with
SD 1.5: NFC normalization -> whitespace squeeze -> lowercase, regex split, byte
level BPE, and Roberta-style start/end tokens with max_length=77 padding.
Requires the `regex` package plus local vocab.json/merges.txt (downloaded when
you first run CLIP/SD via Hugging Face).
"""

import argparse
import csv
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

try:
    import regex as re  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: install the `regex` package (e.g. pip install regex)."
    ) from exc

# Regexes copied from tokenizer.json (see `normalizer` and `Split` pretokenizer).
WHITESPACE_RE = re.compile(r"\s+")
CLIP_SPLIT_RE = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+", re.UNICODE
)


def bytes_to_unicode() -> Dict[int, str]:
    """Map bytes to unique unicode characters (as in OpenAI BPE)."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: Tuple[str, ...]) -> set:
    """Return set of symbol pairs in a word."""
    pairs = set()
    if not word:
        return pairs
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class ClipBPETokenizer:
    """Faithful reproduction of Hugging Face CLIP tokenizer used in SD1.x."""

    def __init__(self, vocab_path: Path, merges_path: Path, max_length: int = 77):
        with vocab_path.open("r", encoding="utf-8") as vf:
            self.encoder: Dict[str, int] = json.load(vf)
        with merges_path.open("r", encoding="utf-8") as mf:
            merges = [
                tuple(line.strip().split())
                for line in mf
                if line.strip() and not line.startswith("#")
            ]
        self.bpe_ranks = {merge: idx for idx, merge in enumerate(merges)}
        self.cache: Dict[str, str] = {}
        self.byte_encoder: Dict[int, str] = bytes_to_unicode()
        self.max_length = max_length
        self.start_token = self.encoder["<|startoftext|>"]
        self.end_token = self.encoder["<|endoftext|>"]

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = WHITESPACE_RE.sub(" ", text)
        return text.lower()

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if (
                    i < len(word) - 1
                    and word[i] == first
                    and word[i + 1] == second
                ):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        merged = " ".join(word)
        self.cache[token] = merged
        return merged

    def _encode_normalized(self, normalized: str) -> List[int]:
        tokens = CLIP_SPLIT_RE.findall(normalized)
        bpe_tokens: List[int] = []
        for token in tokens:
            token_bytes = token.encode("utf-8")
            token = "".join(self.byte_encoder[b] for b in token_bytes)
            for bpe_token in self.bpe(token).split(" "):
                try:
                    bpe_tokens.append(self.encoder[bpe_token])
                except KeyError:
                    # HF tokenizer would map unknown pieces to the end token.
                    bpe_tokens.append(self.end_token)
        return bpe_tokens

    def encode_core(self, text: str) -> List[int]:
        normalized = self.normalize(text)
        return self._encode_normalized(normalized)

    def encode_with_special(self, text: str) -> Tuple[List[int], bool]:
        """Return padded token ids plus whether truncation occurred."""
        tokens = [self.start_token]
        tokens.extend(self.encode_core(text))
        tokens.append(self.end_token)

        truncated = False
        if len(tokens) > self.max_length:
            truncated = True
            tokens = tokens[: self.max_length]
            tokens[-1] = self.end_token
        if len(tokens) < self.max_length:
            tokens.extend([self.end_token] * (self.max_length - len(tokens)))
        return tokens, truncated

    def token_length(self, text: str) -> Tuple[int, bool]:
        """Return (pre-trunc token count incl. special, truncated flag)."""
        core = len(self.encode_core(text))
        total = core + 2  # start + end
        return total, total > self.max_length


def find_tokenizer_dir(explicit: Optional[Path]) -> Path:
    if explicit:
        return explicit
    base = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--openai--clip-vit-large-patch14"
        / "snapshots"
    )
    if not base.exists():
        raise FileNotFoundError(
            f"Tokenizer snapshots not found under {base}. "
            "Run transformers.CLIPTokenizer.from_pretrained once to cache them."
        )
    snapshots = sorted(
        (p for p in base.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not snapshots:
        raise FileNotFoundError(f"No tokenizer snapshots found under {base}.")
    return snapshots[0]


def iter_prompts(path: Path, text_column: Optional[str]) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\r\n")
                if line != "":
                    yield line
        return

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError(f"{path} is missing a header row.")
            column = resolve_column(reader.fieldnames, text_column)
            for row in reader:
                prompt = row.get(column, "")
                if prompt:
                    yield prompt
        return

    raise ValueError(f"Unsupported file extension for {path}")


def resolve_column(headers: List[str], preferred: Optional[str]) -> str:
    if preferred and preferred in headers:
        return preferred
    lower = {name.lower(): name for name in headers}
    for candidate in ("text", "caption", "prompt"):
        if candidate in lower:
            return lower[candidate]
    return headers[0]


@dataclass
class FileStats:
    path: Path
    total: int = 0
    truncated: int = 0
    longest: int = 0
    longest_example: Optional[str] = None


def analyze_file(path: Path, tokenizer: ClipBPETokenizer, column: Optional[str]) -> FileStats:
    stats = FileStats(path=path)
    for prompt in iter_prompts(path, column):
        stats.total += 1
        length, exceeded = tokenizer.token_length(prompt)
        if length > stats.longest:
            stats.longest = length
            stats.longest_example = prompt
        if exceeded:
            stats.truncated += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count CLIP token lengths and truncations for prompt files."
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[
            "MS-COCO_val2014_30k_captions.csv",
            "test.txt",
            "test100.txt",
        ],
        help="Prompt files to inspect (relative to this script by default).",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        help="Directory containing vocab.json and merges.txt. "
        "Defaults to newest cached CLIP snapshot.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        help="CSV column containing prompts; auto-detected if omitted.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=77,
        help="Token limit (Stable Diffusion uses 77).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    tokenizer_dir = find_tokenizer_dir(args.tokenizer_dir)
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"
    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(
            f"Cannot find vocab.json/merges.txt in {tokenizer_dir}. "
            "Ensure CLIP tokenizer assets are present."
        )

    tokenizer = ClipBPETokenizer(vocab_path, merges_path, max_length=args.max_length)

    for file_arg in args.files:
        path = Path(file_arg)
        if not path.is_absolute():
            path = (script_dir / path).resolve()
        if not path.exists():
            print(f"[WARN] {path} not found, skipping.", file=sys.stderr)
            continue
        stats = analyze_file(path, tokenizer, args.text_column)
        print(f"\nFile: {path}")
        print(f"  Prompts scanned:      {stats.total}")
        print(f"  Longest prompt tokens:{stats.longest}")
        print(f"  Truncated prompts:    {stats.truncated} (> {tokenizer.max_length})")
        if stats.longest_example:
            sample = stats.longest_example
            if len(sample) > 120:
                sample = sample[:117] + "..."
            print(f"  Example longest prompt: {sample}")


if __name__ == "__main__":
    main()
