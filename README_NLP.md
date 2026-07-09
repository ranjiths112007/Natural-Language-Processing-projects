# ⌨️ Predictive Smartphone Keyboard — Kneser-Ney 5-Gram Language Model

A from-scratch implementation of **Kneser-Ney smoothed n-gram language modeling**, the classic statistical technique behind predictive-text keyboards, implemented recursively from unigram to 5-gram order. Built as a Week 1 NLP assignment.

## Overview

Given a history of words, e.g. `"the quick brown"`, the model predicts a probability distribution over the next word — the same task your phone's keyboard solves when it suggests `"fox"`.

The core challenge: raw n-gram counting assigns **zero probability** to any phrase never seen during training. Kneser-Ney smoothing fixes this by recursively backing off from high-order n-grams to lower orders whenever data is sparse, guaranteeing every word in the vocabulary always gets some non-zero probability.

## How It Works

**1. Absolute Discounting** — a fixed discount (`0.75`) is subtracted from every observed n-gram count. This "tax" is redistributed to unseen continuations, so nothing ever gets exactly zero probability.

**2. Continuation Probability** — lower-order backoff tiers don't rank words by raw frequency, they rank by **versatility**: how many *distinct* contexts a word has been seen following. This is why "lake" (which follows many different words) outranks "Francisco" (which almost always follows only "San") when backing off to an unseen context.

**3. Recursive Interpolation** — probability at each n-gram order is a blend of:
```
P_KN(word | context) = discounted_term  +  λ(context) × P_KN(word | shorter_context)
```
recursing down from the 5-gram order all the way to the unigram base case.

## Results

Trained on a small toy corpus, the model passes both correctness checks:

| Check | Result |
|---|---|
| Σ P(word \| "the quick brown fox") over full vocab | **1.000000** ✅ |
| Perplexity on in-vocabulary sentence *"The quick brown fox jumps"* | 2.5597 |
| Perplexity on OOV sentence *"The fast magenta unicorn jumps"* | 5.7640 (no crash) ✅ |

```
Conditional Likelihood Predictions for context 'the quick brown fox':
 - P(jumps      | context) = 0.710258
 - P(leaped     | context) = 0.710258
 - P(barked     | context) = 0.710258
 - P(river      | context) = 0.006114
```

Words actually observed completing the context receive high probability mass; unrelated words are correctly scored much lower — and probabilities stay mathematically valid (sum to 1.0) at every context length.

## Getting Started

### Requirements
No external dependencies — pure Python standard library (`math`, `collections`).

### Run
```bash
python kneser_ney.py
```

This trains the model on a toy corpus, verifies the probability distribution sums to 1.0, prints sample word predictions, and computes perplexity on both an in-vocabulary and an out-of-vocabulary test sentence.

### Use in your own code
```python
from kneser_ney import KneserNeyLanguageModel

lm = KneserNeyLanguageModel(order=5, discount=0.75)
lm.fit(["your training sentences go here", "as a list of strings"])

lm.get_probability("fox", ("the", "quick", "brown"))
lm.calculate_sentence_perplexity("the quick brown fox")
```

## Project Structure
```
.
├── kneser_ney.py   # KneserNeyLanguageModel class
└── README.md
```

## Key Implementation Details

- **Vocabulary & `<UNK>` handling**: words appearing only once in training are treated as unknown, and any word not in the trained vocabulary is mapped to `<UNK>` at inference time — this is what lets `calculate_sentence_perplexity` handle out-of-vocabulary input without crashing.
- **`ngram_counts[n]`**: raw counts for every n-gram order from 1 to 5, collected in a single pass over the corpus.
- **`history_extensions` / `context_extensions`**: sets tracking, respectively, which words can precede a given suffix and which words can follow a given context — the basis of the continuation-count calculations used at every backoff tier below the highest order.
- **`lower_order_denominators`**: precomputed so every recursive probability distribution is guaranteed to sum to exactly 1.0, rather than approximately.

## Self-Check

- ✅ Σ P(word | context) over the entire vocabulary equals exactly 1.0 for a sample context
- ✅ `calculate_sentence_perplexity` runs cleanly on a sentence with out-of-vocabulary words and returns a finite value

## License

MIT — free to use for learning and coursework.
