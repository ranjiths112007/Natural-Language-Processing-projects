# Natural Language Processing Projects

A collection of my **NLP coursework, experiments, and task submissions**, built while learning how computers process, represent, and understand human language.

This repository is intentionally practical: each notebook focuses on one NLP idea and turns the concept into working Python code rather than keeping it purely theoretical.

## What I Worked On

- **Hidden Markov Models & Viterbi decoding** — sequence modelling and POS tagging, including a vectorized implementation in NumPy.
- **TF-IDF & Bag of Words** — converting text into numerical features for machine-learning workflows.
- **Word Embeddings** — exploring Word2Vec and document-level representations such as Doc2Vec.
- **Naive Bayes** — applying probabilistic modelling to text classification.
- **Kneser-Ney smoothing** — understanding how language models handle unseen or rare n-grams.
- **Text embeddings and representation techniques** — experiments covering different ways of representing language numerically.

## Repository Structure

```text
Natural-Language-Processing-projects/
├── NLP3.ipynb
├── NLP_Task2_HMM_Viterbi.ipynb
├── NLP_Task7_Doc2Vec.ipynb
├── NLP_Task8_Naive_Bayes.ipynb
├── NLP_Task_4_TF_IDF_BoW.ipynb
├── Task_5/
│   ├── README.md
│   ├── Gutenburg.zip
│   └── Word2Vec.ipynb
├── task6 embedding.ipynb
├── knesar_ney.ipynb
└── README.md
```

## Tech Stack

**Python · NumPy · SciPy · NLP · Machine Learning · Word Embeddings · Probabilistic Models**

## A Bit About the Learning

The main thing I learned from these tasks is that NLP is not just about calling a pretrained model. A lot of the foundation comes from understanding **how text becomes data**, how probabilities behave in language, and how sequential information can be modelled efficiently.

The HMM/Viterbi task was especially useful for connecting the theory of sequence modelling with an implementation that uses vectorized NumPy operations and log-space calculations.

## Running the Notebooks

Most notebooks can be opened directly in **Jupyter Notebook, JupyterLab, or Google Colab**. Install the libraries used by the individual notebook before running it.

```bash
pip install numpy scipy gensim
```

Some notebooks may require additional packages depending on the experiment.

## Coursework

These are primarily **college learning and task-submission projects**. They document my progression from classical NLP techniques toward modern representation and language-modelling concepts.

## Submission

Coursework submission link: https://docs.google.com/forms/d/e/1FAIpQLSe27sXp-jEJ5HbR1U09_9Kz_-v6Vgex0aiSFWw4tHo2EUWWHQ/viewform

---

Built while learning NLP — one task at a time.
