# 🧠 NLProc-Proj-M-SS25: Retrieval-Augmented Generation (RAG) System

Welcome to the official repository for our NLP project: a Retrieval-Augmented Generation (RAG) system that combines dense vector retrieval (FAISS + Sentence Transformers) with generative QA using the FLAN-T5 language model.

🔗 **GitHub Repository:** [https://github.com/HuzaifaAneel09/NLProc-Proj-M-SS25-main](https://github.com/HuzaifaAneel09/NLProc-Proj-M-SS25-main)

---

## 📌 Project Overview

This system takes a natural language question, retrieves relevant facts from a knowledge base (`tech_facts.txt`), and generates an answer using a language model. It is built with modular components that include:

- A **Retriever** for document chunking and semantic search
- A **Generator** for prompt-based answer generation
- A **Pipeline** to combine retrieval and generation
- An **Evaluation script** with logging and test input support

---

## 📂 Directory & File Structure

```
NLPROC-PROJ-M-SS25-MAIN/
│
├── baseline/
│   ├── data/
│   │   └── tech_facts.txt
│   ├── generator/
│   │   └── generator.py
│   ├── retriever/
│   │   ├── retriever.py
│   │   ├── week2_embedding_analysis.py
│   │   ├── week2_faiss_search.py
│   │   └── retriever_store/
│   └── pipeline.py
│
├── evaluation/
│   ├── evaluation.py
│   ├── logs.json
│   └── test_inputs.json
│
├── utils/
│   └── utils.py
│
├── class_1_example/
│   ├── intro.ipynb
│   └── winnie_the_pooh.txt
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 How Vector Search Works

Vector search works by representing both documents and queries as numerical vectors in a high-dimensional space using sentence embeddings. The system measures the **distance or similarity** between these vectors using metrics like **cosine similarity** or **Euclidean (L2) distance**.

For example:
* Similar questions like "Who invented Python?" and "Who created Python?" will have embeddings close together.
* FAISS efficiently searches for nearest neighbors in this vector space to find the most semantically similar document chunks to a given query.

---

## 📊 Observations from FAISS Retrieval

In the FAISS experiment, different phrasings of the question **"Where is the Eiffel Tower?"** consistently retrieve the correct fact about its location. However, **slight shifts in wording can change the rank or distance of retrieval results**.

**Example differences:**
* "Where is the Eiffel Tower?" vs "Tell me the location of the Eiffel Tower."
* All are semantically similar, but due to token or structure differences, the embedding space may slightly shift the result rankings.

This highlights that:
* **Semantic models generalize well**, but are still sensitive to phrasing.
* Preprocessing, paraphrasing, or reranking could help improve robustness.

---

## 📘 File and Function Descriptions

### `baseline/data/tech_facts.txt`
- Contains a collection of short factual statements used as the system's knowledge base.

---

### `baseline/retriever/retriever.py`
Implements the `Retriever` class that manages vector-based document retrieval using Sentence Transformers and FAISS.

**Main Functions:**
- `add_documents(filepath)`: Loads, chunks, embeds, and indexes text.
- `query(question, top_k)`: Retrieves top-k similar chunks to the question.
- `save(path)`: Saves FAISS index and document list.
- `load(path)`: Loads the index and documents.
- `initialize_or_load(data_path, save_path)`: Loads existing index or builds a new one.

---

### `baseline/retriever/week2_embedding_analysis.py`
Encodes 10 manually selected sentences using `all-MiniLM-L6-v2`, computes cosine similarity between them, and visualizes their relationships using PCA.
* **Purpose**: Explore how sentence embeddings capture semantic similarity.
* **Key output**: Heatmap of cosine similarity + 2D scatter plot via PCA.

---

### `baseline/retriever/week2_faiss_query_variants.py`
Demonstrates FAISS-based retrieval by storing 10 sample facts, and querying them with multiple phrasings of the same question (e.g., "Where is the Eiffel Tower?").
* **Purpose**: Understand how small changes in query wording affect similarity search.
* **Key output**: Top-k retrieved facts and their distances for each query.

---

### `baseline/generator/generator.py`
Implements the `Generator` class to build prompts and generate answers using `google/flan-t5-base`.

**Main Functions:**
- `build_prompt(context, question)`: Creates a structured prompt from input.
- `generate_answer(context, question)`: Generates an answer from the language model.

---

### `baseline/pipeline.py`
Defines the `RAGPipeline` class that combines retriever and generator for end-to-end QA.

**Main Functions:**
- `run(question, top_k=3)`: Runs the retrieval and generation steps and returns:
  - question
  - retrieved_chunks
  - prompt
  - generated_answer
  - timestamp
  - group_id

---

### `evaluation/evaluation.py`
Script to run batch testing of the pipeline with logging and result validation.

**Main Functions:**
- `load_test_inputs(filepath)`: Loads test questions from JSON.
- `log_result(result, log_path)`: Appends result to log file.
- `evaluate_pipeline(...)`: Runs tests and prints pass/fail results.

---

### `evaluation/test_inputs.json`
- Contains known QA pairs used for evaluation.
```json
[
  {
    "question": "Who created Python?",
    "expected_answer_contains": "Guido van Rossum"
  },
  ...
]
```

---

### `utils/utils.py`

Provides helper functions for file I/O and text chunking.

**Functions:**

* `load_txt_file(filepath)`: Loads plain text content from a `.txt` file.
* `chunk_text(text, chunk_size=100, overlap=50)`: Splits text into overlapping word chunks for better semantic indexing.

---

### `class_1_example/intro.ipynb` & `winnie_the_pooh.txt`

* Sample notebook and file for exploring RAG behavior with other documents.

---

### `requirements.txt`

Lists required Python packages. Install with:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Run Evaluation

```bash
python evaluation/evaluation.py
```

This will:

* Load the knowledge base
* Run test questions
* Print pass/fail results
* Log answers in `evaluation/logs.jsonl`

---

## ✅ Output Example

```
Q: Who created Python?
Expected: guido van rossum
Got: Guido van Rossum created Python.
✅ PASS
```

---

## 🔄 Ask a Custom Question (Interactive Mode)

You can also ask your own custom question using the same pipeline. This is useful for quick testing or demo purposes without modifying test files.

💬 **Example:**

```bash
python evaluation/evaluation.py --custom_question "Who developed ChatGPT?"
```

This will:
* Load the knowledge base
* Retrieve relevant context
* Generate and display an answer
* Save the result to `evaluation/logs.jsonl`

---

### 📈 Week 2 Experiments (Embeddings & FAISS Retrieval) (How to Run)

**A. Sentence Embedding Similarity & PCA Visualization**

```bash
python baseline/retriever/week2_embedding_analysis.py
```

**B. Query Variants with FAISS Similarity Search**

```bash
python baseline/retriever/week2_faiss_query_variants.py
```

