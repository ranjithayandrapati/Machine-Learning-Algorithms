# 🎯 Supervised Learning Algorithms

Supervised learning is a type of machine learning where we train a model on **labeled data** — meaning each training example comes with an input (`X`) and the correct output (`y`). The model learns a mapping from inputs to outputs and can then predict outcomes for unseen data.

---

## 📚 Key Categories

### 1. **Regression Algorithms**
Used when the output variable is **continuous**.

- **Linear Regression** – predicts a straight-line relationship between features and target.
- **Polynomial Regression** – models non-linear relationships by adding polynomial terms.
- **Ridge / Lasso Regression** – regularized versions to avoid overfitting.
- **Support Vector Regression (SVR)** – uses support vectors for robust regression.
- **Decision Tree Regressor** – tree-based splitting rules for numeric predictions.
- **Random Forest Regressor** – ensemble of decision trees for stable performance.
- **Gradient Boosted Regressors** – boosting approach (XGBoost, LightGBM, CatBoost).

---

### 2. **Classification Algorithms**
Used when the output variable is **categorical**.

- **Logistic Regression** – baseline for binary/multi-class classification.
- **k-Nearest Neighbors (kNN)** – predicts based on closest neighbors.
- **Support Vector Machines (SVM)** – finds hyperplanes to separate classes.
- **Decision Tree Classifier** – rule-based model with splits for each feature.
- **Random Forest Classifier** – ensemble of trees, reduces variance.
- **Gradient Boosted Trees** – powerful boosting approach for tabular data.
- **Naïve Bayes** – probabilistic classifier using Bayes’ theorem.
- **Neural Networks (MLP / Deep Learning)** – learn complex patterns.

---

### 3. **Evaluation Metrics**
How we measure performance depends on the problem type.

- **Regression**: MSE, RMSE, MAE, R² Score  
- **Classification**: Accuracy, Precision, Recall, F1-score, ROC-AUC

---

## 🛠️ Quick Example (Random Forest Classification)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target
)

# Train model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names))

