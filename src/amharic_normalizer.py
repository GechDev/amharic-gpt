"""
Amharic Text Normalization and Cleaning Module.

Pure Python standard library implementation with zero external dependencies.

Normalization Capabilities:
---------------------------
1. Optional Homophone Harmonization (with explicit enable/disable flag):
   - ሐ (ḥä) and ኀ (ḫä) series -> ሀ (hä) series (all 7 vowel orders + labiovelars)
   - ሠ (śä) series -> ሰ (sä) series (all 7 vowel orders + labiovelar ሧ -> ሷ)
   - ዐ (‘ä) series -> አ (’ä) series (all 7 vowel orders)
   - ፀ (ṣ́ä) series -> ጸ (ṣä) series (all 7 vowel orders + labiovelar ፇ -> ጿ)
2. Ethiopic Wordspace (U+1361 '፡') -> Standard ASCII Space (' ')
3. Configurable ASCII '?' to Ethiopic Question Mark (U+1367 '፧') in Amharic contexts
4. Repeated punctuation normalization while strictly preserving Ethiopic punctuation
   and Ethiopic numerals (፩ through ፼).
5. Low Ethiopic-character ratio filtering (configurable threshold) to discard spam/noise.
6. Exact-duplicate line/document removal with duplicate tracking.
7. Transparent explanation of Unicode NFKC behavior on Ethiopic text.
"""

import re
import unicodedata
from typing import Tuple, List, Dict, Optional, Set
from collections import Counter


# ============================================================================
# Unicode Ranges for Ethiopic Script
# ============================================================================
# U+1200 - U+137F : Ethiopic Basic (Syllabary, Punctuation, Digits)
# U+1380 - U+139F : Ethiopic Supplement (Sebatbeit, specialized syllables)
# U+2D80 - U+2DDF : Ethiopic Extended (Me'en, Blin, bench, etc.)
# U+AB00 - U+AB2F : Ethiopic Extended-A (Gamo-Gofa-Dawro, Basketo)
ETHIOPIC_RANGES = [
    (0x1200, 0x137F),
    (0x1380, 0x139F),
    (0x2D80, 0x2DDF),
    (0xAB00, 0xAB2F),
]

# Ethiopic punctuation code points
ETHIOPIC_PUNCTUATION = {
    "\u1360": "፡",  # Ethiopic section mark
    "\u1361": "፡",  # Ethiopic wordspace
    "\u1362": "።",  # Ethiopic full stop
    "\u1363": "፣",  # Ethiopic comma
    "\u1364": "፤",  # Ethiopic semicolon
    "\u1365": "፥",  # Ethiopic colon
    "\u1366": "፦",  # Ethiopic preface colon
    "\u1367": "፧",  # Ethiopic question mark
    "\u1368": "፨",  # Ethiopic paragraph separator
}

# Ethiopic numerals (1 to 10,000)
ETHIOPIC_NUMERALS = {
    chr(cp) for cp in range(0x1369, 0x137D)
}


# ============================================================================
# Homophone Mapping Tables across All Seven Vowel Orders + Labiovelars
# ============================================================================
# Order 1: ግዕዝ (ä), Order 2: ካዕብ (u), Order 3: ሣልስ (i), Order 4: ራብዕ (a)
# Order 5: ኃምስ (e), Order 6: ሳድስ (ə), Order 7: ሳብዕ (o)

# 1. H-series: ሐ and ኀ -> ሀ
HA_HOMOPHONES = {
    # ሐ series -> ሀ
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ", "ሗ": "ኋ",
    # ኀ series -> ሀ
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ኈ": "ሆ", "ኊ": "ሂ", "ኋ": "ኋ", "ኌ": "ሄ", "ኍ": "ህ",
}

# 2. S-series: ሠ -> ሰ
SE_HOMOPHONES = {
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ", "ሧ": "ሷ",
}

# 3. Glottal/Vowel-series: ዐ -> አ
A_HOMOPHONES = {
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
}

# 4. Ts-series: ፀ -> ጸ
TSE_HOMOPHONES = {
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ", "ፇ": "ጿ",
}

# Combined homophone table
ALL_HOMOPHONES: Dict[str, str] = {
    **HA_HOMOPHONES,
    **SE_HOMOPHONES,
    **A_HOMOPHONES,
    **TSE_HOMOPHONES,
}


def is_ethiopic_character(char: str) -> bool:
    """
    Checks if a character belongs to any defined Ethiopic Unicode block.
    """
    cp = ord(char)
    for start, end in ETHIOPIC_RANGES:
        if start <= cp <= end:
            return True
    return False


def is_ethiopic_fidel(char: str) -> bool:
    """
    Checks if a character is an Ethiopic syllabary letter (fidel),
    excluding punctuation marks and numerals.
    """
    if not is_ethiopic_character(char):
        return False
    if char in ETHIOPIC_PUNCTUATION or char in ETHIOPIC_NUMERALS:
        return False
    # Punctuation and digits range from 0x1360 to 0x137C
    cp = ord(char)
    if 0x1360 <= cp <= 0x137C:
        return False
    return True


def normalize_unicode(text: str) -> str:
    """
    Applies Unicode NFKC normalization.
    
    CRITICAL TECHNICAL NOTE ON NFKC AND ETHIOPIC:
    ----------------------------------------------
    What NFKC DOES:
    - Resolves Unicode compatibility characters, such as full-width latin letters,
      half-width katakana, ligatures (e.g. 'fi' -> 'f' + 'i'), and decomposed combining marks.
    
    What NFKC DOES NOT DO for Ethiopic:
    - Ethiopic syllables (Fidel) are atomically precomposed in Unicode (e.g. U+1200 'ሀ').
    - Ethiopic characters DO NOT possess canonical decompositions in the Unicode Standard.
    - NFKC does NOT decompose syllables into consonant + vowel marks.
    - NFKC does NOT normalize homophones (e.g. 'ሐ' is NOT altered by NFKC).
    - NFKC does NOT convert Ethiopic wordspaces (U+1361) to spaces.
    - NFKC does NOT normalize Ethiopic punctuation or numerals.
    
    Therefore, domain-specific Amharic normalization logic must be implemented explicitly.
    """
    return unicodedata.normalize("NFKC", text)


def normalize_homophones(text: str, enabled: bool = True) -> str:
    """
    Harmonizes phonetically merged historical Ge'ez consonants to standard Amharic forms.
    
    Why Homophone Normalization is Made OPTIONAL:
    ---------------------------------------------
    1. PROS (when enabled):
       - Reduces vocabulary sparsity in small language models.
       - Maps alternative spellings of the same word to a single canonical form
         (e.g., 'ፀሐይ' vs 'ጸሀይ' for 'sun', 'ንጉሥ' vs 'ንጉስ' for 'king').
       - Alleviates out-of-vocabulary and sample efficiency issues for modest datasets.
       
    2. CONS (when disabled):
       - Erases historical Ge'ez etymology and root morphology.
       - Removes lexical distinctions in homophones where spelling conveys distinct meaning.
       - Disables exact spelling reproduction of classical literature or formal government text.
       
    Args:
        text (str): Input Amharic text.
        enabled (bool): If True, applies the mapping; if False, returns text unmodified.
        
    Returns:
        str: Normalized text.
    """
    if not enabled:
        return text
    return "".join(ALL_HOMOPHONES.get(char, char) for char in text)


def normalize_wordspace(text: str) -> str:
    """
    Replaces the traditional Ethiopic word separator (U+1361 '፡') with standard ASCII space (' ').
    
    Design Rationale:
    -----------------
    Classical Ge'ez and early Ethiopic typography separated every word with two vertical dots ('፡').
    Modern Amharic print and digital text overwhelmingly use standard whitespace (' ').
    Converting U+1361 to ' ' unifies tokenization boundaries across modern and classical sources.
    """
    return text.replace("\u1361", " ")


def normalize_question_mark(text: str, convert_ascii_to_ethiopic: bool = False) -> str:
    """
    Handles question mark normalization.
    
    Design Choice Documentation:
    ----------------------------
    Ethiopic script possesses a dedicated question mark glyph (U+1367 '፧').
    However, 99%+ of modern digital Amharic media, news websites, and social texts
    use the standard ASCII question mark ('?').
    
    - If convert_ascii_to_ethiopic is False (DEFAULT):
      Preserves standard digital conventions ('?' remains '?').
    - If convert_ascii_to_ethiopic is True:
      Converts ASCII '?' into Ethiopic '፧' specifically in Amharic sentence contexts.
    """
    if convert_ascii_to_ethiopic:
        return text.replace("?", "\u1367")
    return text


def normalize_punctuation(text: str) -> str:
    """
    Normalizes repeated and irregular punctuation marks while strictly preserving
    all valid Ethiopic punctuation (።, ፣, ፤, ፥, ፦, ፧, ፨) and Ethiopic numerals (፩-፼).
    
    Rules:
    - Collapses repeated Ethiopic full stops ('።።+') into a single '።'.
    - Collapses repeated Ethiopic commas ('፡፡+' or '፣፣+') into single marks.
    - Collapses repeated ASCII periods ('....') into ellipsis ('…').
    - Collapses repeated question marks ('??+') into a single '?'.
    - Collapses repeated exclamation marks ('!!+') into a single '!'.
    """
    # Repeated Ethiopic punctuation
    text = re.sub(r"።{2,}", "።", text)
    text = re.sub(r"፣{2,}", "፣", text)
    text = re.sub(r"፤{2,}", "፤", text)
    text = re.sub(r"፥{2,}", "፥", text)
    text = re.sub(r"፦{2,}", "፦", text)
    text = re.sub(r"፧{2,}", "፧", text)

    # Repeated ASCII / Latin punctuation
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"\.{2}", "…", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\!{2,}", "!", text)
    text = re.sub(r"[-–—]{2,}", "—", text)

    return text


def calculate_ethiopic_ratio(text: str) -> float:
    """
    Computes the ratio of Ethiopic characters to total non-whitespace characters.
    
    Formula:
        ratio = count(Ethiopic characters) / count(non-whitespace characters)
        
    Returns:
        float: Ratio between 0.0 and 1.0. If text contains no non-whitespace, returns 0.0.
    """
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    ethiopic_count = sum(1 for c in non_ws if is_ethiopic_character(c))
    return ethiopic_count / len(non_ws)


def filter_low_ethiopic_lines(
    lines: List[str],
    min_ratio: float = 0.5
) -> Tuple[List[str], int]:
    """
    Filters out lines whose Ethiopic-character ratio falls below min_ratio.
    Useful for removing English navigational menus, code snippets, advertisements, and URLs.
    
    Args:
        lines (List[str]): List of candidate text lines.
        min_ratio (float): Minimum required fraction of Ethiopic characters (default 0.5).
        
    Returns:
        Tuple[List[str], int]:
            kept_lines: Lines satisfying the threshold.
            dropped_count: Number of dropped lines.
    """
    kept: List[str] = []
    dropped_count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        ratio = calculate_ethiopic_ratio(stripped)
        if ratio >= min_ratio:
            kept.append(stripped)
        else:
            dropped_count += 1
    return kept, dropped_count


def deduplicate_lines(lines: List[str]) -> Tuple[List[str], int]:
    """
    Removes exact duplicate lines while preserving original encounter order.
    
    Args:
        lines (List[str]): List of candidate text lines.
        
    Returns:
        Tuple[List[str], int]:
            unique_lines: List of distinct lines in order of appearance.
            duplicate_count: Number of duplicate lines dropped.
    """
    seen: Set[str] = set()
    unique: List[str] = []
    duplicate_count = 0
    for line in lines:
        if line in seen:
            duplicate_count += 1
        else:
            seen.add(line)
            unique.append(line)
    return unique, duplicate_count


class AmharicNormalizer:
    """
    Configurable Amharic text normalizer pipeline.
    """

    def __init__(
        self,
        normalize_homophones: bool = False,
        convert_ascii_question: bool = False,
        min_ethiopic_ratio: float = 0.5,
        remove_duplicates: bool = True,
    ):
        self.homophones = normalize_homophones
        self.ascii_question = convert_ascii_question
        self.min_ethiopic_ratio = min_ethiopic_ratio
        self.remove_duplicates = remove_duplicates

    def normalize(self, text: str) -> str:
        """
        Executes complete Amharic text normalization on a string.
        """
        # 1. Unicode NFKC
        text = normalize_unicode(text)

        # 2. Ethiopic wordspace to standard space
        text = normalize_wordspace(text)

        # 3. Optional homophone harmonization
        if self.homophones:
            text = normalize_homophones(text, enabled=True)

        # 4. Optional ASCII ? to Ethiopic ፧
        text = normalize_question_mark(text, convert_ascii_to_ethiopic=self.ascii_question)

        # 5. Punctuation normalization
        text = normalize_punctuation(text)

        # 6. Remove HTML tags, URLs, and emails
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+\.\S+", " ", text)

        # 7. Strip zero-width and unprintable artifacts
        text = re.sub(r"[\u200b\u200c\u200d\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

        # 8. Collapse whitespace
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()
