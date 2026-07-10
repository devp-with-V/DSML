# DSML Portfolio Architecture: AutoInsight & ModelRouter

A comprehensive, dual-pipeline Data Science and Machine Learning framework designed to process flat tabular datasets (CSVs) dynamically. 

## 🌟 Project Overview

This project features two distinct interactive Jupyter Notebook pipelines that pull from a shared, dynamically-registered algorithmic toolbox.

### 1. AutoInsight (Classic Flow)
Located in `autoinsight/notebook.ipynb`, this pipeline represents the traditional data science workflow. 
- You manually select a dataset and algorithm.
- The pipeline profiles the data, executes your chosen model, and generates execution metrics.
- Finally, it uses an LLM to synthesize the results into a visually stunning, custom-themed standalone HTML report.

### 2. ModelRouter (AI-Driven Human-in-the-Loop)
Located in `model_router/notebook.ipynb`, this pipeline automates the data science decision-making process.
- **Smart Profiling:** An LLM intelligently identifies the target variable (or determines if the dataset is unsupervised) purely by reading the column headers.
- **AI Routing:** The dataset profile is sent to an AI Router, which analyzes the data characteristics and recommends the optimal machine learning method.
- **Conversational Reject Loop:** You can accept the AI's recommendation, or click "Reject" to spawn an interactive chat window to interrogate the AI's reasoning. 
- **Inline Visualizations & Premium Reporting:** Upon execution, the pipeline automatically generates inline plots (Confusion Matrices, Scatter Plots), an AI executive summary, and a standalone HTML report.

## 🏗️ Architecture & Registry

The core of the project relies on `shared/registry.py`. Every algorithm in the `shared/methods/` directory uses a `@register_method` decorator.

### Implemented Domains
These algorithms are fully wired to dynamically accept arbitrary tabular CSVs and plot the results:
1. **Classification:** Random Forest, Logistic Regression, SVM, KNN, Naive Bayes, Gradient Boosting, Decision Trees.
2. **Regression:** Linear, Ridge, Lasso, ElasticNet, Polynomial.
3. **Clustering:** K-Means, Hierarchical, DBSCAN, Gaussian Mixture Models.
4. **Dimensionality Reduction:** PCA, t-SNE, UMAP, LDA.
5. **Anomaly Detection:** Isolation Forest, One-Class SVM, Local Outlier Factor.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- An API Key from OpenRouter.ai (for the LLM client)

### Installation
1. Clone the repository.
2. Create a virtual environment and install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your OpenRouter API key:
   ```env
   OPENROUTER_API_KEY=your_api_key_here
   ```
4. Run `python download_datasets.py` to pull sample Kaggle datasets into the `data/` folder.
5. Launch Jupyter Notebook and open either `autoinsight/notebook.ipynb` or `model_router/notebook.ipynb`.
