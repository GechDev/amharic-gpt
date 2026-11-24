#!/usr/bin/env python3
"""
Corpus Ingestion, Normalization, Document Splitting, and Reporting Pipeline.

Features:
---------
1. Ingests raw Amharic text from files or directories in data/raw/ (plain text, JSONL, XML, or archives).
2. Applies domain-specific Amharic normalization:
   - Unicode NFKC (documented compatibility normalization)
   - Ethiopic wordspace (U+1361 '፡') -> standard space
   - Optional homophone harmonization (ሐ/ኀ -> ሀ, ሠ -> ሰ, ዐ -> አ, ፀ -> ጸ across all 7 orders)
   - Optional ASCII '?' -> '፧' conversion
   - Punctuation normalization preserving Ethiopic punctuation and numerals
   - Low-Ethiopic ratio filtering (discards English noise, boilerplate, navigation)
   - Exact-duplicate line/document deduplication with duplicate tracking
3. Document-level seeded deterministic split (90% train, 5% val, 5% test) avoiding contiguous tail bias.
4. Inserts document separator token (<EOS>) between documents.
5. Computes, prints, and saves a comprehensive statistical report (JSON and Markdown).
"""

import os
import sys
import re
import json
import gzip
import bz2
import argparse
from typing import List, Dict, Any, Tuple
from collections import Counter

# Add repository root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.amharic_normalizer import (
    AmharicNormalizer,
    is_ethiopic_character,
    is_ethiopic_fidel,
    calculate_ethiopic_ratio,
    deduplicate_lines,
)
from src.data import split_corpus_documents, generate_corpus_report


def read_raw_corpus(input_path: str) -> List[str]:
    """
    Reads raw documents from a file or directory.
    Supports .txt, .jsonl, .gz, and .bz2 files.
    """
    documents: List[str] = []

    if os.path.isdir(input_path):
        files = sorted(os.listdir(input_path))
        print(f"[Corpus Ingestion] Scanning directory: {input_path} ({len(files)} items)")
        for fname in files:
            fpath = os.path.join(input_path, fname)
            if os.path.isfile(fpath) and not fname.startswith("."):
                documents.extend(read_single_file(fpath))
    elif os.path.isfile(input_path):
        print(f"[Corpus Ingestion] Reading file: {input_path}")
        documents.extend(read_single_file(input_path))
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    return documents


def read_single_file(file_path: str) -> List[str]:
    """
    Reads a single raw text or JSONL file, auto-detecting compression.
    """
    docs: List[str] = []

    if file_path.endswith(".gz"):
        opener = gzip.open(file_path, "rt", encoding="utf-8", errors="replace")
    elif file_path.endswith(".bz2"):
        opener = bz2.open(file_path, "rt", encoding="utf-8", errors="replace")
    else:
        opener = open(file_path, "r", encoding="utf-8", errors="replace")

    with opener as f:
        if file_path.endswith(".jsonl") or file_path.endswith(".jsonl.gz"):
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # Support standard schema keys: 'text', 'content', 'article'
                    text_val = obj.get("text") or obj.get("content") or obj.get("article")
                    if text_val and isinstance(text_val, str) and text_val.strip():
                        docs.append(text_val.strip())
                except json.JSONDecodeError:
                    continue
        else:
            content = f.read()
            # Split by double newline (paragraphs/documents) or single newline if dense
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", content) if p.strip()]
            if len(paragraphs) <= 1:
                paragraphs = [l.strip() for l in content.splitlines() if l.strip()]
            if paragraphs:
                docs.extend(paragraphs)
            elif content.strip():
                docs.append(content.strip())

    return docs


def print_formatted_report(report: Dict[str, Any]) -> None:
    """Prints a clean, human-readable terminal table of corpus statistics."""
    print("=" * 68)
    print("                 AMHARIC CORPUS STATISTICAL REPORT              ")
    print("=" * 68)
    print(f"Total Characters           : {report['total_characters']:,}")
    print(f"Total Words                : {report['total_words']:,}")
    print(f"Average Word Length        : {report['average_word_length']} characters")
    print(f"Unique Characters          : {report['unique_characters']}")
    print(f"Unique Ethiopic Fidel      : {report['unique_fidel_count']}")
    print(f"Unique Ethiopic (All)      : {report['unique_ethiopic_all_count']}")
    print(f"Raw Documents Extracted    : {report['raw_documents_count']:,}")
    print(f"Duplicate Documents Removed: {report['duplicates_removed']:,}")
    print(f"Dropped Low-Ratio Docs     : {report['dropped_low_ratio_documents']:,}")
    print("-" * 68)
    print("Top 20 Characters by Frequency:")
    for rank, item in enumerate(report["top_20_characters"], 1):
        char_display = item["char"] if item["char"] != " " else "<SPACE>"
        print(f"  {rank:>2}. {char_display:>7} : {item['count']:>7,} occurrences ({item['percentage']:>5.2f}%)")
    print("-" * 68)
    print("Corpus Split Partitions:")
    splits = report["splits"]
    for s_name in ["train", "val", "test"]:
        s_data = splits[s_name]
        print(
            f"  {s_name.upper():>5} Split : {s_data['characters']:>9,} chars | "
            f"{s_data['words']:>7,} words | {s_data['documents']:>5,} docs ({s_data['percentage']:>5.2f}%)"
        )
    print("=" * 68)


def save_markdown_report(report: Dict[str, Any], output_path: str) -> None:
    """Writes an artifact-ready Markdown summary of the corpus report."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    splits = report["splits"]
    top_20 = report["top_20_characters"]

    lines = [
        "# Amharic Corpus Statistical Report",
        "",
        "## Summary Metrics",
        "",
        f"- **Total Characters**: {report['total_characters']:,}",
        f"- **Total Words**: {report['total_words']:,}",
        f"- **Average Word Length**: {report['average_word_length']} characters",
        f"- **Total Unique Characters**: {report['unique_characters']}",
        f"- **Unique Ethiopic Fidel**: {report['unique_fidel_count']}",
        f"- **Unique Ethiopic Glyphs (All)**: {report['unique_ethiopic_all_count']}",
        f"- **Raw Ingested Documents**: {report['raw_documents_count']:,}",
        f"- **Exact Duplicate Documents Dropped**: {report['duplicates_removed']:,}",
        f"- **Dropped Low-Ratio Documents**: {report['dropped_low_ratio_documents']:,}",
        "",
        "## Corpus Splits (Seeded Deterministic Split 90/5/5)",
        "",
        "| Split | Characters | Words | Documents | Percentage |",
        "|:---|:---:|:---:|:---:|:---:|",
        f"| **Train** | {splits['train']['characters']:,} | {splits['train']['words']:,} | {splits['train']['documents']:,} | {splits['train']['percentage']}% |",
        f"| **Val**   | {splits['val']['characters']:,}   | {splits['val']['words']:,}   | {splits['val']['documents']:,}   | {splits['val']['percentage']}% |",
        f"| **Test**  | {splits['test']['characters']:,}  | {splits['test']['words']:,}  | {splits['test']['documents']:,}  | {splits['test']['percentage']}% |",
        "",
        "## Top 20 Most Frequent Characters",
        "",
        "| Rank | Character | Unicode Point | Frequency | Percentage |",
        "|:---:|:---:|:---:|:---:|:---:|",
    ]

    for rank, item in enumerate(top_20, 1):
        ch = item["char"]
        display_ch = f"`{ch}`" if ch != " " else "`<SPACE>`"
        hex_cp = f"U+{ord(ch):04X}"
        lines.append(f"| {rank} | {display_ch} | `{hex_cp}` | {item['count']:,} | {item['percentage']}% |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Corpus Report] Saved Markdown summary -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Amharic Corpus Preparation & Ingestion Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/sample_amharic.txt",
        help="Path to raw input file or directory in data/raw/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Target directory for processed splits",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.90,
        help="Training partition ratio (default: 0.90)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Validation partition ratio (default: 0.05)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.05,
        help="Test partition ratio (default: 0.05)",
    )
    parser.add_argument(
        "--min_ethiopic_ratio",
        type=float,
        default=0.5,
        help="Minimum Ethiopic character fraction per document (default: 0.5)",
    )
    parser.add_argument(
        "--normalize_homophones",
        action="store_true",
        help="Harmonize homophones (ሐ/ኀ -> ሀ, ሠ -> ሰ, ዐ -> አ, ፀ -> ጸ)",
    )
    parser.add_argument(
        "--convert_ascii_question",
        action="store_true",
        help="Convert ASCII '?' to Ethiopic '፧' in Amharic text",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for document shuffling",
    )
    parser.add_argument(
        "--doc_separator",
        type=str,
        default="\n<EOS>\n",
        help="Document separator token (default: '\\n<EOS>\\n')",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="outputs",
        help="Directory to save corpus reports",
    )
    args = parser.parse_args()

    # Ingest documents
    raw_documents = read_raw_corpus(args.input)
    raw_corpus_text = "\n\n".join(raw_documents)
    print(f"[Corpus Ingestion] Total raw documents: {len(raw_documents):,}")

    # Split documents with seeded shuffle
    train_text, val_text, test_text, report = split_corpus_documents(
        text=raw_corpus_text,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        doc_separator=args.doc_separator,
        min_ethiopic_ratio=args.min_ethiopic_ratio,
        normalize_homophones_flag=args.normalize_homophones,
        convert_ascii_question=args.convert_ascii_question,
    )

    # Save output splits
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.txt")
    val_path = os.path.join(args.output_dir, "val.txt")
    test_path = os.path.join(args.output_dir, "test.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(val_text)
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(test_text)

    print(f"[Corpus Ingestion] Wrote {train_path} ({len(train_text):,} chars)")
    print(f"[Corpus Ingestion] Wrote {val_path} ({len(val_text):,} chars)")
    print(f"[Corpus Ingestion] Wrote {test_path} ({len(test_text):,} chars)")

    # Print terminal report
    print_formatted_report(report)

    # Save JSON and Markdown reports
    os.makedirs(args.report_dir, exist_ok=True)
    json_path = os.path.join(args.report_dir, "corpus_report.json")
    md_path = os.path.join(args.report_dir, "corpus_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[Corpus Report] Saved JSON metrics -> {json_path}")

    save_markdown_report(report, md_path)


if __name__ == "__main__":
    main()
