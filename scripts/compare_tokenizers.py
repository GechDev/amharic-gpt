#!/usr/bin/env python3
"""
Tokenizer Comparison Benchmark Script.

Compares Character-level Tokenizer vs. BPE Subword Tokenizers (500, 1000, 2000) on Amharic:
- Vocabulary Size
- Total Token Count
- Average Tokens Per Word
- Compression Ratio (relative to character-level)
- Characters Per Token
- Segmentation of 10 Representative Amharic Words
"""

import os
import sys
import json
from typing import List, Dict, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.tokenizer import AmharicTokenizer
from src.bpe_tokenizer import BPETokenizer, WS_MARKER

EXAMPLE_WORDS = [
    "ኢትዮጵያ",
    "ትምህርት",
    "መንግሥት",
    "ዩኒቨርሲቲ",
    "ኮምፒውተር",
    "ሳይንስ",
    "እውቀት",
    "ማህበረሰብ",
    "ባህል",
    "ሰላም",
]


def benchmark_tokenizers(train_text: str, report_dir: str = "outputs") -> Dict[str, Any]:
    words = train_text.split()
    total_words = len(words)
    total_chars = len(train_text)

    configs = [
        {"type": "char", "vocab_size": None, "name": "char"},
        {"type": "bpe", "vocab_size": 500, "name": "bpe-500"},
        {"type": "bpe", "vocab_size": 1000, "name": "bpe-1000"},
        {"type": "bpe", "vocab_size": 2000, "name": "bpe-2000"},
    ]

    results: Dict[str, Any] = {"tokenizers": {}, "word_segmentations": {}}
    char_token_count = 0

    print("=" * 72)
    print("                  AMHARIC TOKENIZER BENCHMARK REPORT             ")
    print("=" * 72)

    for cfg in configs:
        name = cfg["name"]
        if cfg["type"] == "char":
            tok = AmharicTokenizer()
            tok.build_vocab(train_text)
        else:
            tok = BPETokenizer()
            tok.train(train_text, target_vocab_size=cfg["vocab_size"], verbose=False)

        encoded_tokens = tok.encode(train_text)
        total_tokens = len(encoded_tokens)
        if cfg["type"] == "char":
            char_token_count = total_tokens

        tokens_per_word = round(total_tokens / max(1, total_words), 2)
        compression_ratio = round(char_token_count / max(1, total_tokens), 2)
        chars_per_token = round(total_chars / max(1, total_tokens), 2)

        # Segment example words
        word_segments = {}
        for w in EXAMPLE_WORDS:
            if cfg["type"] == "char":
                segments = list(w)
            else:
                raw_segments = tok._encode_word(w)
                segments = [s.replace(WS_MARKER, "") for s in raw_segments]
            word_segments[w] = segments

        results["tokenizers"][name] = {
            "name": name,
            "vocab_size": tok.vocab_size,
            "total_tokens": total_tokens,
            "tokens_per_word": tokens_per_word,
            "compression_ratio": compression_ratio,
            "chars_per_token": chars_per_token,
        }
        results["word_segmentations"][name] = word_segments

        print(
            f"  {name:<10} | Vocab: {tok.vocab_size:>4} | Tokens: {total_tokens:>7,} | "
            f"Tokens/Word: {tokens_per_word:>4.2f} | Compression: {compression_ratio:>4.2f}x | "
            f"Chars/Token: {chars_per_token:>4.2f}"
        )

    print("-" * 72)
    print("Example Amharic Word Segmentations:")
    for w in EXAMPLE_WORDS:
        print(f"\nWord: '{w}'")
        for cfg in configs:
            name = cfg["name"]
            segs = results["word_segmentations"][name][w]
            seg_str = " + ".join(f"[{s}]" for s in segs)
            print(f"  {name:<10}: {seg_str}")
    print("=" * 72)

    # Save JSON and Markdown
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, "tokenizer_comparison.json")
    md_path = os.path.join(report_dir, "tokenizer_comparison.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Benchmark] Saved JSON results -> {json_path}")

    # Generate Markdown Table
    md_lines = [
        "# Amharic Tokenizer Comparison: Character vs. BPE Subwords",
        "",
        "## Quantitative Efficiency Comparison",
        "",
        "| Tokenizer | Vocab Size | Total Tokens | Tokens / Word | Compression Ratio | Chars / Token |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]

    for name, data in results["tokenizers"].items():
        md_lines.append(
            f"| **{name}** | {data['vocab_size']:,} | {data['total_tokens']:,} | "
            f"{data['tokens_per_word']} | {data['compression_ratio']}x | {data['chars_per_token']} |"
        )

    md_lines.extend([
        "",
        "## Segmentation of 10 Representative Amharic Words",
        "",
        "| Amharic Word | Character-Level | BPE-500 | BPE-1000 | BPE-2000 |",
        "|:---|:---:|:---:|:---:|:---:|",
    ])

    for w in EXAMPLE_WORDS:
        char_s = " + ".join(results["word_segmentations"]["char"][w])
        bpe500_s = " + ".join(results["word_segmentations"]["bpe-500"][w])
        bpe1000_s = " + ".join(results["word_segmentations"]["bpe-1000"][w])
        bpe2000_s = " + ".join(results["word_segmentations"]["bpe-2000"][w])
        md_lines.append(f"| **{w}** | `{char_s}` | `{bpe500_s}` | `{bpe1000_s}` | `{bpe2000_s}` |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[Benchmark] Saved Markdown comparison -> {md_path}")

    return results


def main():
    train_path = os.path.join(REPO_ROOT, "data", "processed", "train.txt")
    if not os.path.exists(train_path):
        train_path = os.path.join(REPO_ROOT, "data", "raw", "sample_amharic.txt")
    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()
    benchmark_tokenizers(text)


if __name__ == "__main__":
    main()
