"""
Supervised classification on the Iris dataset
- Pandas loading, Seaborn pairplot
- StandardScaler (fit on train only), 80/20 stratified split
- Logistic Regression vs Decision Tree
- Accuracy / Precision / Recall / F1, confusion matrices, decision boundaries
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # remove this line if running interactively
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

OUT = "results/figures"  # run from the repo root
os.makedirs(OUT, exist_ok=True)
RANDOM_STATE = 42

# ------------------------------------------------------------------
# 1. Load dataset with Pandas
# ------------------------------------------------------------------
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
feature_cols = iris.feature_names
class_names = list(iris.target_names)

print("Shape:", df.shape)
print(df.head(), "\n")
print(df.describe().round(2), "\n")
print("Missing values:\n", df.isna().sum(), "\n")
print("Class balance:\n", df["species"].value_counts(), "\n")

# ------------------------------------------------------------------
# 2. Exploratory pairplot
# ------------------------------------------------------------------
pp = sns.pairplot(df.drop(columns="target"), hue="species", diag_kind="kde",
                  palette="Set2", corner=False)
pp.fig.suptitle("Iris Pairplot", y=1.02)
pp.savefig(os.path.join(OUT, "01_pairplot.png"), dpi=150, bbox_inches="tight")
plt.close("all")

# ------------------------------------------------------------------
# 3. Split (80/20, stratified) and scale
#    Scaler is fit on the TRAIN set only to avoid data leakage.
# ------------------------------------------------------------------
X = df[feature_cols]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print(f"Train: {X_train_s.shape}, Test: {X_test_s.shape}\n")

# ------------------------------------------------------------------
# 4. Train models
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
}

results, preds = [], {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    preds[name] = y_pred
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
    })
    print(f"=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

results_df = pd.DataFrame(results).set_index("Model").round(4)
print("Benchmark summary:\n", results_df, "\n")
results_df.to_csv(os.path.join(OUT, "metrics_summary.csv"))

# Bar chart comparing the metrics
ax = results_df.T.plot(kind="bar", figsize=(8, 5), colormap="Set2", rot=0)
ax.set_ylim(0.5, 1.05)
ax.set_title("Model Benchmark")
ax.set_ylabel("Score")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_metrics_comparison.png"), dpi=150)
plt.close()

# ------------------------------------------------------------------
# 5a. Confusion matrix heatmaps
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, y_pred) in zip(axes, preds.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_confusion_matrices.png"), dpi=150)
plt.close()

# ------------------------------------------------------------------
# 5b. Decision boundaries
#     Boundaries can only be drawn in 2D, so we retrain both models on
#     two features (petal length & petal width) purely for visualization.
# ------------------------------------------------------------------
viz_features = ["petal length (cm)", "petal width (cm)"]
Xv_train = scaler_v = StandardScaler().fit(X_train[viz_features])
Xv_train_s = scaler_v.transform(X_train[viz_features])
Xv_test_s = scaler_v.transform(X_test[viz_features])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (name, model) in zip(axes, models.items()):
    m2d = type(model)(**model.get_params()).fit(Xv_train_s, y_train)
    acc2d = accuracy_score(y_test, m2d.predict(Xv_test_s))

    x_min, x_max = Xv_train_s[:, 0].min() - 0.5, Xv_train_s[:, 0].max() + 0.5
    y_min, y_max = Xv_train_s[:, 1].min() - 0.5, Xv_train_s[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    Z = m2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="Set2")
    sns.scatterplot(x=Xv_train_s[:, 0], y=Xv_train_s[:, 1],
                    hue=[class_names[i] for i in y_train], palette="Set2",
                    edgecolor="k", s=50, ax=ax)
    ax.set_title(f"{name} (2D, test acc = {acc2d:.2f})")
    ax.set_xlabel("petal length (scaled)")
    ax.set_ylabel("petal width (scaled)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_decision_boundaries.png"), dpi=150)
plt.close()

print("Saved figures: 01_pairplot, 02_metrics_comparison, "
      "03_confusion_matrices, 04_decision_boundaries")
