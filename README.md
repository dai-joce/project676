# Amazon Clothing, Shoes, and Jewelry Q&A — Question Clustering & Answer Ambiguity Analysis

This project explores the **Amazon Clothing, Shoes, and Jewelry Q&A** dataset, which contains **22k+ paired customer questions and answers**. The goal is to uncover patterns in how customers ask product questions, investigate what signals **ambiguity** (especially in yes/no questions), and test whether modern NLP embeddings (transformer-based sentence representations) can improve **grouping/retrieval** of semantically similar questions.  

🎥 **Project video:** https://youtu.be/IPUn4PxG8jM

---

## 👉 Start here: `main_notebook.ipynb`
The main deliverable is the curated notebook:
- **`main_notebook.ipynb`**

This is the notebook I used for the final analysis, results, and conclusions.

---

## Research questions
1. How can **clustering** and **advanced language models** be used to automatically group similar customer questions and improve retrieval of clear, helpful answers?
2. What patterns or clusters emerge among the most frequently asked questions for popular products?
3. Which factors are most indicative of **ambiguous answers** to **yes/no** questions?
4. Can **transformer-based** language models improve grouping and retrieval of semantically similar questions compared to traditional clustering approaches?

---

## Data
**Dataset:** Amazon Clothing, Shoes, and Jewelry Q&A  
Source: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/qa/  
Direct download:  
https://mcauleylab.ucsd.edu/public_datasets/data/amazon/qa/qa_Clothing_Shoes_and_Jewelry.json.gz

**What I did (preprocessing):**
- Lowercased questions
- Removed punctuation/normalized tokens (tokenization was used for word frequency analysis)
- Filtered out missing/empty entries
- Used the cleaned question text for TF-IDF and embedding-based clustering

---

## Reproducing the project (Colab workflow)
This project was built/run in **Google Colab**.

### Requirements
At the root of the repo you should have:
- **`requirements.txt`** (exported with `!pip freeze > requirements.txt`)

### Steps to reproduce
In general, follow the order below:

1. Open the notebooks in this order:
   - `checkpoints/project_chkpt1.ipynb` (optional)
   - `checkpoints/project_chkpt2.ipynb` (optional)
   - `main_notebook.ipynb` ✅
2. Ensure your environment matches `requirements.txt`
3. Run the cells top-to-bottom in **`main_notebook.ipynb`**, especially the data load and model sections

---

## Key dependencies
Python version:
- `Python 3.12.13`

Major packages used:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `sentence-transformers` (e.g., `all-MiniLM-L6-v2`)
- `python-louvain`, `igraph` 

The complete list of packages and versions lives in **`requirements.txt`**.

---

## Repo structure

```text
.
├── main_notebook.ipynb
├── requirements.txt
├── .gitignore
├── checkpoints/
│   ├── checkpoint_1.ipynb
│   └── checkpoint_2.ipynb
├── data/
│   └── qa_Clothing_Shoes_and_Jewelry.json.gz

```

---

## Results summary

- **Question type distribution:** The dataset is dominated by **open-ended** questions (**13,878**) compared to **yes/no** questions (**8,190**).

- **Yes/No answer ambiguity:** For `yes/no` questions, the answer type distribution shows frequent ambiguous outcomes, with **`?`** appearing **4,356** times, compared to **`Y` = 2,521** and **`N` = 1,313**.

- **Popular product question trends:** The most frequently asked questions cluster around a small set of products (ASINs). In an example top-10 ASIN analysis, the most-asked ASIN was **B00JL9AY02 (36 questions)**.

- **Text frequency patterns (EDA):** Common tokens in questions include very frequent general words such as **“the”, “i”, “a”, “is”, “size”, “what”**, suggesting that basic lexical cues alone may not fully capture intent without better normalization/feature engineering.

- **Clustering (TF-IDF + KMeans feasibility test):** Using TF-IDF vectors and KMeans on questions from the most frequently asked products, clusters had **uneven sizes** (example K=5 cluster counts: **110, 34, 6, 13, 7**), indicating that some semantic groupings are stronger than others in the popular-product subset.

- **Ambiguity prediction (Logistic Regression):** A logistic regression baseline trained to predict whether a `yes/no` answer is ambiguous (`?`) using simple features like **question length** and **product frequency** achieved **~0.56 accuracy** (classification report shown in the notebook).

- **Transformer embeddings (Sentence-BERT):** I also tested clustering with transformer-based embeddings using `sentence-transformers` (e.g., **`all-MiniLM-L6-v2`**) to represent questions semantically, then applied KMeans to discover embedding-based question clusters for comparison with TF-IDF clustering.

---

## Citations
- Mengting Wan, Julian McAuley, "Modeling ambiguity, subjectivity, and diverging viewpoints in opinion question answering systems", ICDM 2016.
- Julian McAuley, Alex Yang, "Addressing complex and subjective product-related queries with customer reviews", WWW 2016.
