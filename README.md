# NLP Task 2 — Vectorized Hidden Markov Model & Viterbi Decoding for POS Tagging

## Objective
Build a Part-of-Speech (POS) tagger using a Hidden Markov Model (HMM), with the
Viterbi decoding step implemented using **fully vectorized NumPy matrix
operations** (no inner loop over tags), numerical underflow handled via
**log-space transformation**, and tagging **precision evaluated on complex
multi-word sentences**.

## Tech Stack
- Python 3.10+
- NumPy
- SciPy (`logsumexp`, used for a probability sanity check)

## Concept

| Term | Meaning |
|---|---|
| **Transition probability** | `P(tag_i \| tag_{i-1})` — likelihood one tag follows another |
| **Emission probability** | `P(word_i \| tag_i)` — likelihood a tag generates a given word |
| **Goal** | Find the tag sequence `T` maximizing `P(T\|W)`, equivalent to maximizing `P(W\|T) * P(T)` |
| **Brute force** | Checking every tag sequence is `O(k^n)` for `n` words and `k` tags — infeasible even for short sentences |
| **Viterbi (DP)** | Builds a table where each cell `[tag, position]` stores the *best* probability of reaching that tag at that position, reusing sub-results instead of recomputing |
| **Log-space** | Multiplying many small probabilities underflows to 0 in floating point; working in log-space turns multiplication into addition, avoiding this |

## What Makes This "Vectorized"

The baseline version of this algorithm loops over every current tag `j` at each
timestep and computes its best incoming score with a Python `for` loop. This
implementation removes that inner loop and replaces it with a single matrix
operation per timestep:

```python
M = dp[:, t - 1][:, None] + log_trans        # (N, N): every from-tag -> to-tag pair at once
M = M + log_emit[:, word][None, :]            # broadcast emission over columns
dp[:, t]          = M.max(axis=0)             # best score landing on each tag
backpointer[:, t] = M.argmax(axis=0)          # which previous tag produced it
```

This turns an `O(N)` Python loop per timestep into one `(N, N)` NumPy
broadcasting operation per timestep — the core requirement of the task.

## Model Setup
- **Tags (6):** `DET, NOUN, VERB, ADJ, ADV, PREP`
- **Vocabulary:** 28 words covering determiners, nouns, verbs, adjectives,
  adverbs, and prepositions, enough to form grammatically complex sentences
  (e.g. `"a big cat chases the small mouse"`).
- **Initial, transition, and emission probabilities** are hand-set to reflect
  realistic English grammar patterns (e.g. determiners are usually followed
  by nouns or adjectives; verbs are usually followed by adverbs, nouns, or
  prepositional phrases).

## Evaluation
Five hand-labeled complex sentences are run through the vectorized Viterbi
decoder and compared against gold-standard tag sequences to compute
**token-level tagging precision**:

```
TOKEN-LEVEL TAGGING PRECISION: 30/30 = 100.00%
```

## How to Run
```bash
pip install numpy scipy
python vectorized_hmm_viterbi.py
```

## Files
- `vectorized_hmm_viterbi.py` — full implementation (model setup, vectorized
  Viterbi, evaluation harness)
- `README.md` — this file
