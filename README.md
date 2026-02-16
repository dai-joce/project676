# Amazon Clothing, Shoes, and Jewelry Q&A Data Mining Project

## Overview
This project explores the Amazon Clothing, Shoes, and Jewelry Q&A dataset, which contains over 22,000 paired customer questions and answers about products. The goal is to uncover patterns, improve answer quality, and experiment with advanced text mining and NLP models.

## Dataset
- **Source:** [Amazon Q&A data (Julian McAuley, UCSD)]([https://mcauleylab.ucsd.edu/public_datasets.html](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/qa/))
- **Category:** Clothing, Shoes, and Jewelry
- **Entries:** 22,068 Q&A pairs
- **Attributes:** Product ID (ASIN), question type, answer type, timestamps, question text, answer text

## Project Goals
- Perform exploratory data analysis (EDA) to understand question and answer patterns
- Identify product popularity trends and ambiguous answers
- Apply clustering and transformer-based NLP models to improve answer reliability and retrieval

## Usage
1. Download the dataset (see `download_dataset.py`).
2. Load and preprocess the data with Python (see `eda.ipynb`).
3. Run EDA and experiments.

## Requirements
- Python 3.6+
- pandas, numpy, matplotlib, seaborn

## Citations
- Mengting Wan, Julian McAuley, "Modeling ambiguity, subjectivity, and diverging viewpoints in opinion question answering systems", ICDM 2016.
- Julian McAuley, Alex Yang, "Addressing complex and subjective product-related queries with customer reviews", WWW 2016.
