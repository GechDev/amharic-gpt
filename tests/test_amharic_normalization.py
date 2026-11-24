"""
Comprehensive Unit Tests for Amharic Normalization and Preprocessing.

Tests:
1. Homophone normalization across all seven vowel orders and labiovelars
   (ሐ/ኀ -> ሀ, ሠ -> ሰ, ዐ -> አ, ፀ -> ጸ) with real Amharic examples.
2. Optional homophone flag (verifies disabled mode preserves classical spellings).
3. Ethiopic wordspace (U+1361 '፡') converted to standard ASCII space (' ').
4. ASCII '?' conversion to Ethiopic '፧' as a documented design choice.
5. Repeated punctuation normalization while preserving Ethiopic punctuation and numerals.
6. Low-Ethiopic character ratio line filtering.
7. Exact-duplicate line/document removal and duplicate tracking.
8. Deterministic document/paragraph-level seeded split (90/5/5).
9. Document separator token (<EOS>) insertion.
10. Explanation of Unicode NFKC on Ethiopic text.
"""

import unicodedata
import numpy as np
import pytest

from src.amharic_normalizer import (
    AmharicNormalizer,
    normalize_unicode,
    normalize_homophones,
    normalize_wordspace,
    normalize_question_mark,
    normalize_punctuation,
    calculate_ethiopic_ratio,
    filter_low_ethiopic_lines,
    deduplicate_lines,
    is_ethiopic_character,
    is_ethiopic_fidel,
    HA_HOMOPHONES,
    SE_HOMOPHONES,
    A_HOMOPHONES,
    TSE_HOMOPHONES,
)
from src.data import split_corpus_documents, clean_amharic_text


def test_homophone_normalization_all_orders():
    """Verify all 7 vowel orders for Ha, Se, A, and Tse series."""
    # 1. H-series: ሐ and ኀ -> ሀ
    ha_original = "ሐሑሒሓሔሕሖ ኀኁኂኃኄኅኆ"
    ha_expected = "ሀሁሂሃሄህሆ ሀሁሂሃሄህሆ"
    assert normalize_homophones(ha_original, enabled=True) == ha_expected

    # 2. S-series: ሠ -> ሰ
    se_original = "ሠሡሢሣሤሥሦ ሧ"
    se_expected = "ሰሱሲሳሴስሶ ሷ"
    assert normalize_homophones(se_original, enabled=True) == se_expected

    # 3. A-series: ዐ -> አ
    a_original = "ዐዑዒዓዔዕዖ"
    a_expected = "አኡኢኣኤእኦ"
    assert normalize_homophones(a_original, enabled=True) == a_expected

    # 4. Tse-series: ፀ -> ጸ
    tse_original = "ፀፁፂፃፄፅፆ ፇ"
    tse_expected = "ጸጹጺጻጼጽጾ ጿ"
    assert normalize_homophones(tse_original, enabled=True) == tse_expected

    # Real Amharic words
    words_input = "ፀሐይ ንጉሥ ዓለም ሕይወት"
    words_expected = "ጸሀይ ንጉስ ኣለም ህይወት"
    assert normalize_homophones(words_input, enabled=True) == words_expected


def test_homophone_normalization_optional_flag():
    """Verify that disabling homophone normalization preserves original text identically."""
    classical_text = "ፀሐይ በንጉሥ ዐዋጅ ታበራለች፤ የሕይወት ምንጭ ናት።"
    # When disabled (default design):
    result_disabled = normalize_homophones(classical_text, enabled=False)
    assert result_disabled == classical_text

    # In AmharicNormalizer with normalize_homophones=False:
    normalizer = AmharicNormalizer(normalize_homophones=False)
    cleaned = normalizer.normalize(classical_text)
    assert "ፀሐይ" in cleaned
    assert "ንጉሥ" in cleaned
    assert "የሕይወት" in cleaned

    # When enabled:
    normalizer_enabled = AmharicNormalizer(normalize_homophones=True)
    cleaned_enabled = normalizer_enabled.normalize(classical_text)
    assert "ጸሀይ" in cleaned_enabled
    assert "ንጉስ" in cleaned_enabled
    assert "የህይወት" in cleaned_enabled


def test_ethiopic_wordspace():
    """Verify Ethiopic wordspace (U+1361 '፡') is normalized to standard space."""
    classical_sentence = "ሰላም፡ለኢትዮጵያ፡ሀገራችን፡ይሁን።"
    normalized = normalize_wordspace(classical_sentence)
    assert normalized == "ሰላም ለኢትዮጵያ ሀገራችን ይሁን።"
    assert "\u1361" not in normalized


def test_ascii_question_mark_design_choice():
    """Verify ASCII '?' preservation by default and conversion to '፧' when chosen."""
    amharic_question = "እንደምን ዋልክ?"
    
    # Default design choice: preserves digital standard ASCII '?'
    default_res = normalize_question_mark(amharic_question, convert_ascii_to_ethiopic=False)
    assert default_res == "እንደምን ዋልክ?"

    # Explicit design choice: converts to traditional Ethiopic question mark (U+1367 '፧')
    converted_res = normalize_question_mark(amharic_question, convert_ascii_to_ethiopic=True)
    assert converted_res == "እንደምን ዋልክ፧"
    assert "\u1367" in converted_res


def test_repeated_punctuation_and_ethiopic_preservation():
    """Verify normalization of repeated punctuation while preserving Ethiopic punctuation and numerals."""
    dirty_punct = "ሰላም.... እንደምን ነህ???? ይሁን።።።። እናም፣፣፣ እውነት!! ፩፱፺፰ ዓ.ም."
    cleaned_punct = normalize_punctuation(dirty_punct)

    # Ellipsis normalized
    assert "…" in cleaned_punct
    assert "...." not in cleaned_punct

    # Single question mark
    assert "????" not in cleaned_punct
    assert "?" in cleaned_punct

    # Single Ethiopic full stop
    assert "።።" not in cleaned_punct
    assert "።" in cleaned_punct

    # Single Ethiopic comma
    assert "፣፣" not in cleaned_punct
    assert "፣" in cleaned_punct

    # Single exclamation
    assert "!!" not in cleaned_punct
    assert "!" in cleaned_punct

    # Ethiopic numerals preserved intact
    assert "፩፱፺፰" in cleaned_punct


def test_ethiopic_ratio_filtering():
    """Verify line dropping based on low Ethiopic character ratio."""
    lines = [
        "ይህ ንጹህ የአማርኛ ጽሑፍ ነው፤ በጣም ግልጽ እና የሚያምር ነው።",  # Ratio ~1.0
        "English copyright © 2026 All rights reserved.",           # Ratio 0.0
        "የኢትዮጵያ ዜና News Update: Read more at http://site.com",     # Mixed
        "አዲስ አበባ ዩኒቨርሲቲ የትምህርት ማዕከል ነው።",                # Ratio ~1.0
    ]

    kept, dropped = filter_low_ethiopic_lines(lines, min_ratio=0.5)
    assert len(kept) == 2
    assert dropped == 2
    assert "ይህ ንጹህ የአማርኛ ጽሑፍ ነው፤ በጣም ግልጽ እና የሚያምር ነው።" in kept
    assert "አዲስ አበባ ዩኒቨርሲቲ የትምህርት ማዕከል ነው።" in kept


def test_exact_duplicate_removal():
    """Verify exact duplicate removal with duplicate count tracking."""
    lines = [
        "ሰላም ለኢትዮጵያ።",
        "ትምህርት ወሳኝ ነው።",
        "ሰላም ለኢትዮጵያ።",  # Duplicate 1
        "እውቀት ኃይል ነው።",
        "ትምህርት ወሳኝ ነው።",  # Duplicate 2
        "ሰላም ለኢትዮጵያ።",  # Duplicate 3
    ]

    unique, dup_count = deduplicate_lines(lines)
    assert len(unique) == 3
    assert dup_count == 3
    assert unique == ["ሰላም ለኢትዮጵያ።", "ትምህርት ወሳኝ ነው።", "እውቀት ኃይል ነው።"]


def test_deterministic_document_level_split():
    """Verify seeded document/paragraph-level split (90/5/5) is deterministic and reproducible."""
    # Create 20 distinct Amharic documents
    documents = [f"ይህ ሰነድ ቁጥር {i} ነው፤ ስለ ኢትዮጵያ የተለያዩ ርዕሰ ጉዳዮች ይገልጻል።" for i in range(20)]
    corpus_text = "\n\n".join(documents)

    train1, val1, test1, rep1 = split_corpus_documents(
        corpus_text,
        train_ratio=0.90,
        val_ratio=0.05,
        test_ratio=0.05,
        seed=42,
    )

    train2, val2, test2, rep2 = split_corpus_documents(
        corpus_text,
        train_ratio=0.90,
        val_ratio=0.05,
        test_ratio=0.05,
        seed=42,
    )

    # Deterministic with same seed
    assert train1 == train2
    assert val1 == val2
    assert test1 == test2
    assert rep1 == rep2

    # Verify split document counts: 20 docs -> 18 train, 1 val, 1 test
    assert rep1["splits"]["train"]["documents"] == 18
    assert rep1["splits"]["val"]["documents"] == 1
    assert rep1["splits"]["test"]["documents"] == 1

    # Verify different seed produces different split assignment
    train_diff, _, _, _ = split_corpus_documents(
        corpus_text,
        train_ratio=0.90,
        val_ratio=0.05,
        test_ratio=0.05,
        seed=999,
    )
    assert train1 != train_diff, "Different seeds must produce different permutations."


def test_document_separator_insertion():
    """Verify <EOS> document separator is inserted between documents in splits."""
    docs = [
        "አንደኛ ሰነድ ስለ ታሪክ።",
        "ሁለተኛ ሰነድ ስለ ሳይንስ።",
        "ሦስተኛ ሰነድ ስለ ትምህርት።",
        "አራተኛ ሰነድ ስለ ቴክኖሎጂ።",
    ]
    corpus_text = "\n\n".join(docs)

    train_text, val_text, test_text, _ = split_corpus_documents(
        corpus_text,
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=42,
        doc_separator="\n<EOS>\n",
    )

    assert "<EOS>" in train_text
    parts = train_text.split("\n<EOS>\n")
    assert len(parts) == 2


def test_nfkc_on_ethiopic_truth():
    """Verify technically that NFKC does NOT alter Ethiopic Fidel or homophones."""
    sample = "ሀሁ ሐሑ ኀኁ ሠሡ ሰሱ አኡ ዐዑ ጸጹ ፀፁ ፡ ። ፣ ፤ ፥ ፦ ፧ ፩፪፫"
    normalized = unicodedata.normalize("NFKC", sample)
    assert sample == normalized, "NFKC must not decompose or alter precomposed Ethiopic syllables."
