# 🎛️ HyperParameter Tuning — A Complete Learning Guide

Squeeze the best performance out of your ML models through systematic optimization.

---

## 📌 Key Strategies Covered
* **Manual vs. Automated:** Moving from intuition to systematic searches.
* **GridSearchCV:** Exhaustive search across every combination in a grid.
* **RandomizedSearchCV:** Sample-efficient random combinations (often beats Grid Search).
* **Bayesian Optimization:** Using surrogate models to intelligently pick the next set of parameters.
* **Optuna:** Advanced TPE-based framework with built-in pruning and visualization.

## 🌲 Priority Tuning Order
1. **Learning Rate & N_Estimators:** The most impactful pair for boosting models.
2. **Tree Constraints:** Max depth, min samples split, and subsampling.
3. **Regularization:** reg_alpha (L1) and reg_lambda (L2) to prevent overfitting.

## 🚀 Quick Implementation: Optuna
```python
import optuna

def objective(trial):
    # Suggest values for parameters[cite: 5]
    lr = trial.suggest_float('learning_rate', 1e-4, 0.5, log=True)
    depth = trial.suggest_int('max_depth', 1, 20)
    # Return CV score here...
