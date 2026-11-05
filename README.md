# Datheon Machine Learning & Data Science Internship

Machine learning projects completed for the Datheon ML & Data Science internship syllabus. Each task is a self-contained Jupyter notebook that follows the suggested approach for that task.

## Tasks completed

| # | Task | Level | Key techniques |
|---|------|-------|----------------|
| 1 | Multi-Class Benchmark Classification (Iris) | Beginner | StandardScaler, 80/20 split, Logistic Regression, Decision Tree, confusion matrix, decision boundaries |
| 2 | Housing Price Prediction (California Housing) | Beginner | Imputation, feature engineering, VIF / multicollinearity, OLS vs Ridge vs Lasso, GridSearchCV, residual analysis |
| 3 | Customer Churn Classifier (Telco) | Intermediate | OneHotEncoder, SMOTE, Random Forest, XGBoost, ROC-AUC, feature importance |

## Repository structure

```
.
├── notebooks/
│   ├── task1_iris_classification.ipynb
│   ├── task2_housing_regression.ipynb
│   └── task3_churn_classification.ipynb
├── src/
│   └── task1_iris_classification.py   # script version of Task 1
├── data/                              # place the starter datasets here (see data/README.md)
├── results/
│   ├── figures/                       # saved plots from the Task 1 script
│   └── task1_metrics_summary.csv
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone <this-repo-url>
cd <repo-folder>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

Task 1 needs no download because Iris ships with scikit-learn. Tasks 2 and 3 read the starter files from `data/`; see `data/README.md` for the expected file names.

To run the Task 1 script instead of the notebook: `python src/task1_iris_classification.py` (from the repo root).

## Task 1 results (Iris, 80/20 stratified split, `random_state=42`)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| Decision Tree | 0.9333 | 0.9333 | 0.9333 | 0.9333 |

Precision, recall and F1 are macro-averaged. Setosa is classified perfectly; all errors are between versicolor and virginica. Results, plots and metrics for Tasks 2 and 3 are in their notebooks.

## Methodology notes

- Scalers and imputers are fit on training data only, to avoid data leakage.
- In Task 3, SMOTE runs inside an `imblearn` pipeline so only training folds are oversampled.
- Decision boundary plots (Task 1) use two features (petal length and width) because boundaries can only be drawn in 2D.

## Tools

Python, pandas, NumPy, scikit-learn, Seaborn, Matplotlib, XGBoost, imbalanced-learn.
