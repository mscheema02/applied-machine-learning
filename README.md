# Applied Machine Learning — Classification Pipelines

This project showcases **end-to-end supervised classification** workflows across multiple datasets, including a **cost-sensitive risk/default classifier** where the goal is to minimize **business cost** (not just error rate).

---

## Table of Contents

- [Project Highlights](#project-highlights)
- [Repository Contents](#repository-contents)
- [Skills Demonstrated](#skills-demonstrated)
- [Results and Conclusions](#results-and-conclusions)
  - [Q1 — Decision Tree Tuning (Lymph)](#q1--decision-tree-tuning-lymph)
  - [Q2 — Preprocessing + Model Comparison (Credit Approval)](#q2--preprocessing--model-comparison-credit-approval)
  - [Q3 — 10-Fold CV Benchmarking (Ionosphere)](#q3--10-fold-cv-benchmarking-ionosphere)
  - [Q4 — Multi-Dataset Evaluation (UCI via ucimlrepo)](#q4--multi-dataset-evaluation-uci-via-ucimlrepo)
  - [Q5 — Cost-Sensitive Risk Classification (Default Prediction)](#q5--cost-sensitive-risk-classification-default-prediction)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Notes](#notes)

---

## Project Highlights

- Built and compared multiple classifiers using **hold-out testing** + **cross-validation**.
- Implemented robust preprocessing for mixed feature types:
  - missing-value handling (including **IterativeImputer**),
  - categorical encoding (one-hot),
  - scaling with sparse matrices,
  - dimensionality reduction (**TruncatedSVD**).
- Handled **class imbalance** using **SMOTE**.
- Designed and optimized a **custom cost metric** (high penalty for false negatives) for real-world risk classification.
- Produced a final prediction file in the required format: `ORDER-ID CLASS`.

---

## Repository Contents

These are the file names used in this repo:

| File | What it is |
|---|---|
| `classification.ipynb` | Main notebook containing solutions for Q1–Q5 (code + outputs) |
| `cost_sensitive_risk_classification_report.pdf` | Written report for Q5 describing methodology and model selection |
| `datasets.zip` | All datasets used across the assignment |
| `prediction.txt` | Final Q5 prediction output (`ORDER-ID CLASS`) |

---

## Skills Demonstrated

**Modeling**
- Decision Trees, KNN, Gaussian Naive Bayes, Logistic Regression, MLP Neural Networks, Random Forests

**Evaluation**
- Accuracy, error rate, K-Fold CV / Stratified CV
- Confusion-matrix-driven **cost-sensitive scoring**

**Preprocessing / Feature Engineering**
- Mixed numeric + categorical preprocessing
- Iterative and simple imputations
- One-hot encoding + train/test alignment
- Sparse-aware scaling (`StandardScaler(with_mean=False)`)
- Dimensionality reduction (`TruncatedSVD`)
- Date feature extraction (year/month/day)

**Imbalanced Learning**
- Oversampling with SMOTE (and comparison against ADASYN)

---

## Results and Conclusions

### Q1 — Decision Tree Tuning (Lymph)
**What I did**
- Trained a Decision Tree classifier and visualized the tree.
- Demonstrated how changing a key hyperparameter impacts performance.

**Key outcome**
- With `criterion='entropy'` and `min_samples_split=4`, accuracy was **0.8833**
- Increasing `min_samples_split` to **32** reduced accuracy to **0.7667**

**Conclusion**
- This illustrates how controlling split constraints can reduce overfitting *but may also underfit* if set too aggressively.

---

### Q2 — Preprocessing + Model Comparison (Credit Approval)
**Models evaluated**
- Decision Tree
- KNN (k=1), KNN (k=3)
- Gaussian Naive Bayes
- Logistic Regression
- MLP Neural Network
- Random Forest

**Key outcomes (from the notebook run)**
- Best **test accuracy**: **MLP** (~**0.8841**)
- Best **cross-validation accuracy**: **Random Forest** (~**0.8712**)

**Conclusion**
- Different evaluation strategies can change “the best model.”
- Cross-validation provides a more stable estimate than a single hold-out split.

---

### Q3 — 10-Fold CV Benchmarking (Ionosphere)
**What I did**
- Benchmarked multiple models using **10-fold CV** and reported both accuracy and error rate.

**Key outcome**
- Top performers were extremely close:
  - **Random Forest** accuracy ~**0.9317** (best in the run)
  - **MLP** accuracy ~**0.9316** (very close second)

**Conclusion**
- Ensemble methods and neural nets can perform similarly well depending on data structure and features.

---

### Q4 — Multi-Dataset Evaluation (UCI via ucimlrepo)
**Datasets evaluated**
- Balance-scale, Ecoli, Glass, Ionosphere, Iris, Wine, Yeast

**Key outcome**
Ranking by **lowest average error rate** across datasets:
1. **Random Forest** — **0.1474** (best overall)
2. **MLP Neural Network** — **0.1492**
3. **Logistic Regression** — **0.1743**
4. **KNN (k=3)** — **0.1937**
5. **KNN (k=1)** — **0.2020**
6. **Decision Tree** — **0.2177**
7. **GaussianNB** — **0.2774**

**Conclusion**
- Random Forest provided the most consistently strong performance across diverse datasets.

---

### Q5 — Cost-Sensitive Risk Classification (Default Prediction)
“Real-world” part of the assignment: the evaluation target is **business cost**, not accuracy.

#### Pipeline summary
- **Feature engineering:** date parsing into `year/month/day`
- **Imputation:**
  - Numeric → IterativeImputer
  - Categorical → most frequent
- **Encoding:** one-hot encoding (with alignment of train/test columns)
- **Scaling:** sparse-aware standardization (`with_mean=False`)
- **Dimensionality reduction:** TruncatedSVD to reduce one-hot dimensionality
- **Imbalance handling:** SMOTE (ADASYN was tested but crashed in experimentation)
- **Model selection:** cross-validation using a custom cost metric

#### Custom cost function
- False Negative (miss a risky order): **cost = 50**
- False Positive (flag a safe order): **cost = 5**

#### Model comparison (average cost)
- DecisionTree: **25759.00**
- KNN (k=1): **5721.00** ✅ **best**
- KNN (k=3): **8777.00**
- GaussianNB: **31923.00**
- LogisticRegression: **27997.00**
- MLP: **91078.00**
- RandomForest: **27715.00**

**Final model (based on cost):** **KNN (k=1)** with average cost **5721.00**

#### Final prediction output
- Output file: `prediction.txt`
- Class distribution in the produced prediction:
  - **yes:** 3163
  - **no:** 16837

**Conclusion**
- A model with a good overall error rate is not necessarily best when false negatives are expensive.
- Cost-sensitive selection led to a different “best model” than accuracy-based selection.

---

## How to Run

### 1) Setup (Python environment)
Recommended: create a virtual environment (Python 3.x):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -U pip
pip install numpy pandas matplotlib scikit-learn imbalanced-learn ucimlrepo jupyter
```

### 2) Unzip datasets
Unzip `datasets.zip` into a folder such as `data/`:

```bash
mkdir -p data
unzip datasets.zip -d data
```

### 3) Open the notebook
Launch Jupyter:

```bash
jupyter notebook
```

Open:

- `classification.ipynb`

### 4) Update dataset paths
The notebook was originally run in Google Colab and may reference paths like:

- `/content/drive/MyDrive/EECS4412/A3/...`

For local execution, update paths to match your extracted folder, for example:

- `data/lymph.csv`
- `data/credit-a-train.csv`
- `data/credit-a-test.csv`
- `data/ionosphere.csv`
- `data/mapped_risk_train.csv`
- `data/mapped_risk_test.csv`

Tip: use Find/Replace to update the base path in one pass.

### 5) Generate predictions
Run the notebook top-to-bottom. The Q5 section produces a predictions file in:

```
ORDER-ID CLASS
<id1> <yes/no>
<id2> <yes/no>
...
```

---

## Technologies Used

- **Python 3**
- **Jupyter Notebook / Google Colab**
- **NumPy** (numerical computing)
- **Pandas** (data manipulation and feature engineering)
- **scikit-learn** (models, preprocessing, CV, metrics)
- **imbalanced-learn** (SMOTE resampling)
- **Matplotlib** (visualization)
- **ucimlrepo** (UCI dataset retrieval for Q4)

---

## Notes

- Exact metric values can vary slightly depending on library versions and randomness seeds.
- For reproducibility, the notebook sets random seeds in multiple places (where supported).
