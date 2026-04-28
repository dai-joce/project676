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

## Citations
- Mengting Wan, Julian McAuley, "Modeling ambiguity, subjectivity, and diverging viewpoints in opinion question answering systems", ICDM 2016.
- Julian McAuley, Alex Yang, "Addressing complex and subjective product-related queries with customer reviews", WWW 2016.
