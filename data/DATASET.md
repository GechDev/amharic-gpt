# Amharic Dataset Documentation

## 1. Dataset Overview

This project is configured to use natural Amharic text corpora for educational language modeling.

- **Recommended Primary Corpus**: **Amharic News Dataset / BBC News Amharic / Wikipedia Amharic**
- **Public Repositories**:
  - [Amharic-News-Text-Classification-Dataset](https://github.com/IsraelAbebe/Amharic-News-Text-Classification-Dataset) (Israel Abebe Azime & Nebil Mohammed)
  - [Hugging Face Datasets: amharic_news / oscar-corpus / cc100](https://huggingface.co/datasets/mc4)
  - [Amharic Wikipedia Dumps](https://dumps.wikimedia.org/amwiki/)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0) / Public Domain / Open Access
- **Language**: Amharic (አማርኛ) - Ethiopic / Ge'ez script (Unicode U+1200 - U+137F)
- **Sample Corpus**: A bundled curated corpus of Amharic prose covering news, education, history, science, and everyday conversation is provided at `data/raw/sample_amharic.txt` (exact size: 3,940 characters, 741 words, 71 lines, 170 unique characters) for immediate out-of-the-box local verification and educational experiments.

## 2. Obtaining Larger Datasets

To download and train on larger corpora:

1. **Option A (GitHub Open Amharic News Dataset)**:
   ```bash
   # Download the Amharic News dataset (uncompressed ~20MB of natural Amharic articles):
   curl -o data/raw/amharic_corpus.txt https://raw.githubusercontent.com/IsraelAbebe/Amharic-News-Text-Classification-Dataset/master/data/raw/news.txt
   ```

2. **Option B (Custom Amharic Text)**:
   Place any UTF-8 encoded text file with Amharic articles into `data/raw/` (e.g. `data/raw/corpus.txt`).

## 3. Preprocessing Pipeline

The preprocessing pipeline in `src/data.py` performs:
1. **Unicode NFKC Normalization**: Resolves composite characters, compatibility forms, and ensures canonical Unicode representation.
2. **Noise and Tag Removal**: Strips HTML tags, email addresses, URLs, zero-width spaces (`\u200b`), and non-printable control characters.
3. **Fidel & Punctuation Preservation**: Preserves the complete Ethiopic fidel syllabary (`\u1200`–`\u137F`), Ethiopic punctuation (`፡`, `።`, `፣`, `፤`, `፥`, `፦`, `፧`, `፨`), Arabic numerals, and standard punctuation marks (`?`, `!`, `,`, `.`, `"`, `'`, `(`, `)`).
4. **Whitespace Regularization**: Consolidates multiple consecutive spaces or redundant empty lines while keeping paragraph structure intact.
5. **Train / Validation Splitting**: Splits the clean corpus chronologically/sequentially into training (90%) and validation (10%) splits (`data/processed/train.txt` and `data/processed/val.txt`).
