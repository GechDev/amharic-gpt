# Small GPT-Style Language Model for Amharic

An educational, from-scratch implementation of an autoregressive GPT-style Transformer language model for the Amharic language, built using Python and pure NumPy.

## Project Structure

```
amharic-gpt/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── checkpoints/
├── outputs/
├── notebooks/
│   └── demo.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── tokenizer.py
│   ├── data.py
│   ├── math_utils.py
│   ├── embeddings.py
│   ├── activations.py
│   ├── normalization.py
│   ├── attention.py
│   ├── feed_forward.py
│   ├── transformer_block.py
│   ├── model.py
│   ├── loss.py
│   ├── optimizer.py
│   ├── train.py
│   ├── generate.py
│   └── utils.py
│
└── tests/
    ├── test_tokenizer.py
    ├── test_attention.py
    ├── test_model.py
    └── test_training.py
```
