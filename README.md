# 🎛️ HyperParameter Tuning — A Complete Learning Guide

> **Master the art and science of optimizing machine learning models through hyperparameter tuning.**  
> This repository serves as a comprehensive reference for students, data scientists, and ML engineers who want to deeply understand how to squeeze the best performance out of any model.

---

## 📌 Table of Contents

1. [What are Hyperparameters?](#1-what-are-hyperparameters)
2. [Hyperparameters vs Parameters](#2-hyperparameters-vs-parameters)
3. [Why Hyperparameter Tuning Matters](#3-why-hyperparameter-tuning-matters)
4. [Common Hyperparameters by Model Type](#4-common-hyperparameters-by-model-type)
5. [Tuning Strategies](#5-tuning-strategies)
   - [Manual Search](#51-manual-search)
   - [Grid Search](#52-grid-search-cv)
   - [Random Search](#53-random-search-cv)
   - [Bayesian Optimization](#54-bayesian-optimization)
   - [Halving Search](#55-halving-search-successive-halving)
   - [Optuna (Advanced)](#56-optuna-advanced-framework)
6. [Cross-Validation in Tuning](#6-cross-validation-in-tuning)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Overfitting vs Underfitting](#8-overfitting-vs-underfitting)
9. [Practical Workflow](#9-practical-workflow)
10. [Code Examples](#10-code-examples)
11. [Tips & Best Practices](#11-tips--best-practices)
12. [Common Mistakes to Avoid](#12-common-mistakes-to-avoid)
13. [Resources & References](#13-resources--references)

---

## 1. What are Hyperparameters?

In machine learning, a **hyperparameter** is a configuration variable that is set **before** the training process begins. Unlike regular model parameters (like weights and biases), hyperparameters are **not learned from the data** — they are defined by the practitioner.

Think of it this way:

> 🏗️ *Parameters* are what the model learns during training (e.g., the weights in a neural network).  
> 🎛️ *Hyperparameters* are the settings you configure before handing the data to the model (e.g., how many layers, how fast to learn).

### Examples of hyperparameters:
- **Learning rate** — How large are the steps taken during gradient descent?
- **Max depth** — How deep should a decision tree grow?
- **Number of estimators** — How many trees in a Random Forest?
- **C and gamma** — Regularization and kernel parameters for SVMs
- **Batch size** — How many samples per training update in a neural network?
- **Number of neighbors (k)** — In KNN, how many neighbors to consider?

---

## 2. Hyperparameters vs Parameters

| Feature | Parameters | Hyperparameters |
|---|---|---|
| Learned from data? | ✅ Yes | ❌ No |
| Set before training? | ❌ No | ✅ Yes |
| Examples | Weights, biases | Learning rate, depth, C |
| Optimized by? | Gradient descent / backprop | Tuning strategies (Grid, Random, Bayesian) |
| Stored in model? | ✅ Yes | Usually separate (in config) |

---

## 3. Why Hyperparameter Tuning Matters

Choosing the wrong hyperparameters can make even the best algorithm perform poorly. The same Random Forest on the same dataset can go from **65% accuracy to 92% accuracy** just by tuning the hyperparameters correctly.

### The stakes:
- 🎯 **Better accuracy** — A well-tuned model generalizes better to unseen data
- ⚡ **Faster training** — Right batch size and learning rate = faster convergence
- 🧠 **Avoid overfitting/underfitting** — Regularization hyperparameters control model complexity
- 💡 **Model selection** — Tuning helps you fairly compare different algorithms
- 🏆 **Competitive edge** — In Kaggle and industry, tuning often separates good from great models

---

## 4. Common Hyperparameters by Model Type

### 🌲 Decision Trees & Random Forests
```
max_depth         — Maximum depth of each tree (prevents overfitting)
n_estimators      — Number of trees in the forest
min_samples_split — Minimum samples required to split a node
min_samples_leaf  — Minimum samples at a leaf node
max_features      — Number of features to consider for the best split
```

### 🚀 Gradient Boosting (XGBoost, LightGBM, CatBoost)
```
learning_rate     — Shrinkage rate (lower = more robust but slower)
n_estimators      — Number of boosting rounds
max_depth         — Depth of each tree
subsample         — Fraction of samples per tree
colsample_bytree  — Fraction of features per tree
reg_alpha         — L1 regularization (sparsity)
reg_lambda        — L2 regularization (smoothness)
```

### 📐 Support Vector Machines (SVM)
```
C                 — Regularization strength (higher = less regularization)
kernel            — 'linear', 'rbf', 'poly', 'sigmoid'
gamma             — Kernel coefficient (for 'rbf', 'poly', 'sigmoid')
degree            — Degree of polynomial kernel
```

### 🧠 Neural Networks (Keras / PyTorch)
```
learning_rate     — Step size for weight updates
batch_size        — Samples per gradient update
epochs            — Number of full dataset passes
dropout_rate      — Fraction of neurons to randomly drop
optimizer         — 'adam', 'sgd', 'rmsprop', etc.
hidden_units      — Number of neurons per layer
activation        — 'relu', 'tanh', 'sigmoid', etc.
```

### 👥 KNN (K-Nearest Neighbors)
```
n_neighbors       — Number of neighbors to consider
weights           — 'uniform' or 'distance'
metric            — 'euclidean', 'manhattan', 'minkowski'
```

### 📊 Logistic Regression
```
C                 — Inverse regularization strength
solver            — 'lbfgs', 'saga', 'liblinear', etc.
penalty           — 'l1', 'l2', 'elasticnet', 'none'
max_iter          — Maximum iterations for convergence
```

---

## 5. Tuning Strategies

### 5.1 Manual Search

The simplest method — you manually try different combinations based on intuition and domain knowledge.

**When to use:**
- Very limited compute
- You already have prior knowledge about reasonable ranges
- Quick sanity checks

**Drawbacks:**
- Time-consuming
- Subjective
- Hard to be systematic

---

### 5.2 Grid Search CV

**GridSearchCV** exhaustively searches every combination in a defined parameter grid.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,                    # 5-fold cross validation
    scoring='accuracy',
    n_jobs=-1,               # Use all CPU cores
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
```

**Pros:**
- Exhaustive — never misses the best combo within the grid
- Reproducible

**Cons:**
- Computationally expensive — tries ALL combinations
- Doesn't scale well with many hyperparameters (curse of dimensionality)

> 💡 **Rule of thumb:** If you have 3 hyperparameters with 5 values each = 5³ = **125 combinations** × CV folds. Use with care.

---

### 5.3 Random Search CV

**RandomizedSearchCV** samples a fixed number of random combinations from the parameter space.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100,              # Try 100 random combinations
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=2
)

random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)
```

**Why RandomSearch often beats GridSearch:**

Research by Bergstra & Bengio (2012) showed that **random search finds better hyperparameters in less time** because not all hyperparameters are equally important, and random sampling covers the space more efficiently.

**Pros:**
- Faster than Grid Search
- Searches continuous distributions
- Scales better with many hyperparameters

**Cons:**
- Does not guarantee finding the optimal combination
- Results vary per run (use `random_state` for reproducibility)

---

### 5.4 Bayesian Optimization

Instead of random or exhaustive search, **Bayesian Optimization** builds a probabilistic model of the objective function and uses it to intelligently decide the next set of hyperparameters to try.

It works in a loop:
1. Train a **surrogate model** (usually Gaussian Process) on past evaluations
2. Use an **acquisition function** (e.g., Expected Improvement) to suggest the next point
3. Evaluate the actual model with those hyperparameters
4. Update the surrogate model
5. Repeat

```python
# Using scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

search_space = {
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(1, 30),
    'learning_rate': Real(0.001, 0.5, prior='log-uniform'),
    'subsample': Real(0.5, 1.0)
}

bayes_search = BayesSearchCV(
    estimator=GradientBoostingClassifier(),
    search_spaces=search_space,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

bayes_search.fit(X_train, y_train)
print("Best Parameters:", bayes_search.best_params_)
```

**Pros:**
- Much more sample-efficient than Grid or Random Search
- Leverages past evaluations to guide future search
- Excellent for expensive model evaluations (e.g., deep learning)

**Cons:**
- More complex to implement
- Surrogate model overhead for very cheap evaluations
- Can get stuck in local optima

---

### 5.5 Halving Search (Successive Halving)

A smart resource-efficient strategy that trains many candidates with a **small budget** first, then progressively gives more resources to the best performers.

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

halving_search = HalvingRandomSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=param_dist,
    factor=3,                # Keep top 1/3 of candidates each round
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

halving_search.fit(X_train, y_train)
print("Best Params:", halving_search.best_params_)
```

**Pros:**
- Very fast — eliminates poor candidates early
- Great for large datasets and many hyperparameters

---

### 5.6 Optuna (Advanced Framework)

**Optuna** is a state-of-the-art hyperparameter optimization framework that uses **Tree-structured Parzen Estimator (TPE)** — a sophisticated Bayesian approach.

```python
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
    }
    
    model = GradientBoostingClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return score.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, timeout=600)

print("Best Trial:", study.best_trial.params)
print("Best Score:", study.best_trial.value)

# Visualize
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

**Why Optuna stands out:**
- Supports **pruning** (stop bad trials early)
- Beautiful built-in visualizations
- Supports **distributed optimization** across multiple machines
- Works with any ML framework (sklearn, XGBoost, PyTorch, TensorFlow)

---

## 6. Cross-Validation in Tuning

**Cross-validation (CV)** is essential during hyperparameter tuning to get a reliable estimate of model performance and prevent data leakage.

### K-Fold Cross Validation
```
Dataset → Split into K folds
For each fold:
    Train on K-1 folds
    Validate on remaining 1 fold
Average all K validation scores = CV score
```

```python
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print(f"CV Score: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Stratified K-Fold (for classification)
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Ensures class balance in each fold
```

> ⚠️ **Important:** Always tune hyperparameters using only the **training set**. The test set must remain completely unseen until the very end.

---

## 7. Evaluation Metrics

Choose your scoring metric wisely — it directly guides what the tuning optimizes for.

| Task | Metric | When to use |
|---|---|---|
| Classification (balanced) | `accuracy` | Equal class distribution |
| Classification (imbalanced) | `f1`, `roc_auc` | Unequal class distribution |
| Classification (rare positives) | `precision`, `recall` | Fraud detection, medical diagnosis |
| Regression | `neg_mean_squared_error` | Standard regression |
| Regression | `neg_mean_absolute_error` | When outliers shouldn't dominate |
| Regression | `r2` | Explained variance |

```python
# Using different scoring metrics
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='f1_weighted',   # Change this based on your problem
    cv=5
)
```

---

## 8. Overfitting vs Underfitting

Understanding this is crucial before tuning:

```
          High Bias              Optimal              High Variance
         (Underfitting)                              (Overfitting)
              |___________________________|___________________________|
         Simple model                                   Complex model
         Low train score                                High train score
         Low test score                                 Low test score
```

### Diagnosing with Learning Curves

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
```

### Regularization Hyperparameters

| Problem | Solution |
|---|---|
| Overfitting | Increase regularization (C↓ in SVM, alpha↑ in Ridge, dropout↑ in NN) |
| Underfitting | Decrease regularization, increase model complexity |

---

## 9. Practical Workflow

Follow this structured approach every time:

```
1. DEFINE THE PROBLEM
   └── Classification / Regression / Clustering?
   └── Choose a primary evaluation metric

2. BASELINE MODEL
   └── Train with default hyperparameters
   └── Record baseline CV score

3. IDENTIFY KEY HYPERPARAMETERS
   └── Focus on the 2-4 most impactful ones first
   └── Refer to model documentation

4. COARSE SEARCH
   └── Use RandomizedSearchCV with wide ranges
   └── Goal: find promising regions quickly

5. FINE SEARCH
   └── Use GridSearchCV or Bayesian around best region
   └── Narrow the search space

6. FINAL EVALUATION
   └── Train final model with best params on full training set
   └── Evaluate ONCE on held-out test set

7. DOCUMENT & SAVE
   └── Save best parameters to config file
   └── Record all results for reproducibility
```

---

## 10. Code Examples

### Full End-to-End Tuning Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV,
    GridSearchCV, cross_val_score
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data ──────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 2. Create Pipeline ────────────────────────────────
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(random_state=42))
])

# ── 3. Baseline Score ─────────────────────────────────
baseline = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
print(f"Baseline F1: {baseline.mean():.4f} ± {baseline.std():.4f}")

# ── 4. Coarse Random Search ───────────────────────────
param_dist = {
    'model__n_estimators': randint(50, 500),
    'model__max_depth': randint(1, 10),
    'model__learning_rate': uniform(0.01, 0.3),
    'model__subsample': uniform(0.6, 0.4),
    'model__min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=100,
    cv=5, scoring='f1', random_state=42,
    n_jobs=-1, verbose=1
)
random_search.fit(X_train, y_train)
print("\nCoarse Best Params:", random_search.best_params_)
print(f"Coarse Best F1: {random_search.best_score_:.4f}")

# ── 5. Fine Grid Search ───────────────────────────────
best = random_search.best_params_
fine_grid = {
    'model__n_estimators': [
        max(50, best['model__n_estimators'] - 50),
        best['model__n_estimators'],
        best['model__n_estimators'] + 50
    ],
    'model__learning_rate': [
        round(best['model__learning_rate'] * 0.5, 4),
        round(best['model__learning_rate'], 4),
        round(best['model__learning_rate'] * 1.5, 4)
    ],
    'model__max_depth': [
        max(1, best['model__max_depth'] - 1),
        best['model__max_depth'],
        best['model__max_depth'] + 1
    ]
}

grid_search = GridSearchCV(
    pipeline, fine_grid, cv=5,
    scoring='f1', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)
print("\nFine Best Params:", grid_search.best_params_)
print(f"Fine Best F1: {grid_search.best_score_:.4f}")

# ── 6. Final Evaluation ───────────────────────────────
final_model = grid_search.best_estimator_
y_pred = final_model.predict(X_test)
print("\n" + "="*50)
print("FINAL TEST SET PERFORMANCE")
print("="*50)
print(classification_report(y_test, y_pred, target_names=data.target_names))
```

### Saving and Loading Best Parameters

```python
import json

# Save best parameters
best_params = grid_search.best_params_
with open('best_hyperparams.json', 'w') as f:
    json.dump(best_params, f, indent=2)

# Load and reuse
with open('best_hyperparams.json', 'r') as f:
    loaded_params = json.load(f)

# Remove pipeline prefix for direct model use
model_params = {k.replace('model__', ''): v for k, v in loaded_params.items()}
final_model = GradientBoostingClassifier(**model_params, random_state=42)
```

---

## 11. Tips & Best Practices

### ✅ Do's
- **Always use cross-validation** — never tune on the test set
- **Start with Random Search** — it's more efficient than Grid Search for most cases
- **Tune the most impactful hyperparameters first** — use domain knowledge and documentation
- **Use a Pipeline** — prevents data leakage and makes code clean
- **Log all experiments** — use MLflow, W&B, or even a simple CSV
- **Set random_state** — ensures reproducibility
- **Use `n_jobs=-1`** — parallelize across all CPU cores
- **Consider the compute budget** — Bayesian optimization for expensive models

### 🔢 Hyperparameter Importance Order (General)

For most tree-based models, tune in this priority:
1. `learning_rate` + `n_estimators` (most impactful)
2. `max_depth` / `max_leaves`
3. `subsample` / `colsample_bytree`
4. Regularization (`reg_alpha`, `reg_lambda`)
5. Other minor parameters

---

## 12. Common Mistakes to Avoid

| ❌ Mistake | ✅ Correct Approach |
|---|---|
| Tuning on the test set | Use cross-validation on training data only |
| Searching too wide a grid | Start broad, then narrow down |
| Ignoring preprocessing in pipeline | Always include scaler/encoder in pipeline |
| Not setting `random_state` | Always set for reproducibility |
| Over-tuning (too many iterations) | Watch for diminishing returns |
| Using accuracy for imbalanced data | Use F1, ROC-AUC, or PR-AUC |
| Tuning all hyperparameters at once | Focus on the most impactful ones first |
| Forgetting to retrain on full training data | After tuning, retrain on all of `X_train` |

---

## 13. Resources & References

### 📚 Books
- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* — Aurélien Géron
- *The Elements of Statistical Learning* — Hastie, Tibshirani, Friedman
- *Automated Machine Learning* — Frank Hutter et al.

### 🌐 Documentation
- [Scikit-Learn: Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Scikit-Optimize](https://scikit-optimize.github.io/)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)

### 📄 Research Papers
- Bergstra & Bengio (2012) — *Random Search for Hyper-Parameter Optimization*
- Snoek et al. (2012) — *Practical Bayesian Optimization of Machine Learning Algorithms*
- Li et al. (2017) — *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization*

---

## 👨‍💻 Author

**Syed Faizan Ali**  
Computer Science Student | AI/ML Engineer 
📍 Rahim Yar Khan, Punjab, Pakistan  

[![GitHub](https://img.shields.io/badge/GitHub-SyedFaizanAlii-black?logo=github)](https://github.com/SyedFaizanAlii)
[![Kaggle](https://img.shields.io/badge/Kaggle-syedfaizanalii-blue?logo=kaggle)](https://www.kaggle.com/syedfaizanalii)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://in/syed-faizan-ali-70b164306)

---

> ⭐ *If this repository helped you learn about hyperparameter tuning, please give it a star!*  
> 🍴 *Feel free to fork and contribute improvements.*

---

*Last Updated: 2026 | License: MIT*
