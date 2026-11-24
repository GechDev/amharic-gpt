# Amharic Dataset Documentation & Preprocessing Guide

## 1. Dataset Overview

This project implements an end-to-end data pipeline tailored specifically for the Amharic language (አማርኛ), written in the Ethiopic / Ge'ez script (`\u1200`–`\u137F`).

### 1.1 Smoke-Test Fixture (`data/raw/sample_amharic.txt`)
- **Purpose**: A small, authentic smoke-test fixture designed for fast local unit testing, rapid debugging, and overfitting sanity checks on CPU.
- **True Provenance**: Curated Amharic educational prose covering 71 discrete statements across Ethiopian history, education, science, culture, and computing.
- **Exact Metrics**:
  - Raw characters: **3,940**
  - Cleaned characters: **4,009** (including document separators)
  - Words: **741**
  - Lines / Documents: **71**
  - Unique characters: **170**
  - Unique Ethiopic Fidel: **162**
- **License**: Creative Commons Zero (CC0) / Public Domain dedication for educational demonstration.

---

## 2. Real Large Amharic Corpora (5 MB to 20+ MB)

For substantial model training, `scripts/prepare_corpus.py` ingests, cleans, deduplicates, and splits real Amharic corpora placed in `data/raw/`:

### Supported Public Sources & Licenses:
1. **Amharic Wikipedia Dump (`amwiki`)**:
   - **Source**: Wikimedia Downloads (`amwiki-latest-pages-articles.xml.bz2`)
   - **Size**: ~15 MB compressed, ~45 MB extracted text.
   - **License**: Creative Commons Attribution-ShareAlike 3.0 / 4.0 (CC BY-SA 3.0/4.0).
   - **Ingestion**: Place extracted text or raw XML dump into `data/raw/amwiki.txt` or `data/raw/`.

2. **MasakhaNEWS Amharic**:
   - **Source**: Masakhane NLP benchmark for African news languages.
   - **Size**: ~5 MB to 12 MB of clean Amharic news articles across politics, business, and health.
   - **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
   - **Ingestion**: Place JSONL or plain text export into `data/raw/masakhanews_amharic.jsonl`.

3. **CC-100 Amharic Monolingual Corpus ("am")**:
   - **Source**: Common Crawl monolingual subset processed by Facebook AI (`am.txt.xz`).
   - **Size**: ~20 MB to 150 MB.
   - **License**: Common Crawl Terms of Use / Research and Educational Use.
   - **Ingestion**: Place decompressed `am.txt` or `am.txt.gz` into `data/raw/`.

> [!NOTE]
> *Download URL Protocol*: Per university assignment rules, URLs are not hardcoded or automatically scraped. When a specific remote dump URL is provided by the instructor or user, it is fetched into `data/raw/` and processed using `scripts/prepare_corpus.py`.

---

## 3. What Unicode NFKC Does and Does NOT Do for Ethiopic

Many machine learning projects mistakenly assume that standard Unicode NFKC normalization "handles Amharic text". Here is the exact technical truth according to the Unicode Standard:

### What NFKC DOES:
- Normalizes compatibility glyphs: full-width ASCII characters, half-width katakana, ligatures (e.g. `fi` ligature -> `f` + `i`), and superscripts (`²` -> `2`).
- Composes decomposed Latin/Greek base letters with combining diacritics into canonical precomposed characters.

### What NFKC DOES NOT DO for Ethiopic Script:
1. **No Syllable Decomposition**: Ethiopic syllables (Fidel) are atomically precomposed in Unicode (e.g. U+1200 `ሀ`, U+1201 `ሁ`). Ethiopic glyphs do not possess canonical decompositions in the Unicode Character Database. NFKC leaves every Fidel syllable 100% untouched.
2. **No Homophone Merging**: NFKC does not map phonetically equivalent letters (e.g. `ሐ` and `ኀ` remain distinct from `ሀ`).
3. **No Wordspace Normalization**: The traditional Ethiopic word separator (U+1361 `፡`) is not converted to an ASCII space by NFKC.
4. **No Punctuation Normalization**: Ethiopic full stops (`።`), commas (`፣`), and colons (`፥`) are not regularized by NFKC.

Because NFKC does not perform these operations, our custom `AmharicNormalizer` module (`src/amharic_normalizer.py`) implements each step explicitly.

---

## 4. Normalization Rules & Design Choices (`src/amharic_normalizer.py`)

### 4.1 Optional Homophone Normalization (`--normalize_homophones`)
In modern spoken Amharic, several historical Ge'ez consonants have merged phonetically:
- **Ha series**: `ሐ` (ḥä) and `ኀ` (ḫä) -> `ሀ` (hä)
- **Se series**: `ሠ` (śä) -> `ሰ` (sä)
- **A series**: `ዐ` (‘ä) -> `አ` (’ä)
- **Tse series**: `ፀ` (ṣ́ä) -> `ጸ` (ṣä)

All seven vowel orders (ግዕዝ, ካዕብ, ሣልስ, ራብዕ, ኃምስ, ሳድስ, ሳብዕ) plus labiovelars are mapped:

| Series | Order 1 (ä) | Order 2 (u) | Order 3 (i) | Order 4 (a) | Order 5 (e) | Order 6 (ə) | Order 7 (o) | Labiovelar |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ሀ** (canonical) | ሀ | ሁ | ሂ | ሃ | ሄ | ህ | ሆ | ኋ |
| **ሐ** (mapped) | ሐ -> ሀ | ሑ -> ሁ | ሒ -> ሂ | ሓ -> ሃ | ሔ -> ሄ | ሕ -> ህ | ሖ -> ሆ | ሗ -> ኋ |
| **ኀ** (mapped) | ኀ -> ሀ | ኁ -> ሁ | ኂ -> ሂ | ኃ -> ሃ | ኄ -> ሄ | ኅ -> ህ | ኆ -> ሆ | ኈ -> ሆ, etc. |
| **ሰ** (canonical) | ሰ | ሱ | ሲ | ሳ | ሴ | ስ | ሶ | ሷ |
| **ሠ** (mapped) | ሠ -> ሰ | ሡ -> ሱ | ሢ -> ሲ | ሣ -> ሳ | ሤ -> ሴ | ሥ -> ስ | ሦ -> ሶ | ሧ -> ሷ |
| **አ** (canonical) | አ | ኡ | ኢ | ኣ | ኤ | እ | ኦ | - |
| **ዐ** (mapped) | ዐ -> አ | ዑ -> ኡ | ዒ -> ኢ | ዓ -> ኣ | ዔ -> ኤ | ዕ -> እ | ዖ -> ኦ | - |
| **ጸ** (canonical) | ጸ | ጹ | ጺ | ጻ | ጼ | ጽ | ጾ | ጿ |
| **ፀ** (mapped) | ፀ -> ጸ | ፁ -> ጹ | ፂ -> ጺ | ፃ -> ጻ | ፄ -> ጼ | ፅ -> ጽ | ፆ -> ጾ | ፇ -> ጿ |

- **Why Optional?**
  - *When Enabled*: Reduces vocabulary sparsity and unifies alternate spellings (`ፀሐይ` vs `ጸሀይ` for "sun", `ንጉሥ` vs `ንጉስ` for "king").
  - *When Disabled (Default)*: Preserves historical Ge'ez etymology, formal government spelling, and morphological distinctions.

### 4.2 Ethiopic Wordspace (U+1361 `፡`) -> Space (`" "`)
Classical texts use two vertical dots (`፡`) between words. Digital text uses ASCII whitespace (`" "`). Converting `፡` to `" "` unifies token boundaries across all eras of Amharic prose.

### 4.3 ASCII `?` vs Ethiopic `፧` Design Choice
Ethiopic script defines a question mark glyph (U+1367 `፧`). However, modern digital Amharic media overwhelmingly uses ASCII `?`.
- **Default**: Preserves digital ASCII `?`.
- **Optional (`--convert_ascii_question`)**: Converts ASCII `?` to `፧` for classical typography experiments.

### 4.4 Repeated Punctuation & Fidel Preservation
- Repeated punctuation is collapsed (`።።+` -> `።`, `....` -> `…`, `????` -> `?`, `!!!!` -> `!`).
- All Ethiopic punctuation marks (`።`, `፣`, `፤`, `፥`, `፦`, `፧`, `፨`) and Ethiopic numerals (`፩` through `፼`) are strictly preserved.

### 4.5 Low Ethiopic-Character Ratio Filtering
Web dumps frequently contain English copyright notices, navigational links, and code snippets.
- Formula: $\text{ratio} = \frac{\text{count}(\text{Ethiopic characters})}{\text{count}(\text{non-whitespace characters})}$
- Lines or documents where $\text{ratio} < 0.5$ (configurable via `--min_ethiopic_ratio`) are filtered out.

### 4.6 Exact-Duplicate Line and Document Removal
Duplicate headlines, repetitive navigation elements, and boilerplate paragraphs are dropped to prevent memorization artifacts during training. Duplicate counts are recorded in the corpus report.

---

## 5. Document-Level Seeded Splitting & Document Separators

### 5.1 Why Never Use a Contiguous Tail Split?
Splitting text by slicing the final 10% of characters causes severe distribution shift: if articles are ordered chronologically or grouped by category, the validation set will reflect only the final topic (e.g., all sports). 
`split_corpus_documents()` performs a **seeded pseudo-random permutation** across complete documents/paragraphs, guaranteeing:
1. Identical topic distribution across train, validation, and test splits.
2. Complete documents without cutting sentences in half.
3. Deterministic reproducibility via `--seed 42`.

### 5.2 Document Separator (`<EOS>`)
Between documents, the token `\n<EOS>\n` is inserted into `train.txt`, `val.txt`, and `test.txt`. This enables the autoregressive language model to learn document boundaries and naturally terminate generation when an article concludes.

---

## 6. How to Run the Pipeline

```bash
# Ingest and prepare the smoke-test fixture or any large raw file:
python scripts/prepare_corpus.py --input data/raw/sample_amharic.txt --output_dir data/processed --seed 42

# Or prepare a large raw corpus placed in data/raw/amwiki.txt:
# python scripts/prepare_corpus.py --input data/raw/amwiki.txt --output_dir data/processed --seed 42

# View the generated statistical reports:
cat outputs/corpus_report.md
```
