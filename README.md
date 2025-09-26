# Machine-Learning-Algorithms

# Supervised Algorithms

This folder contains Jupyter notebooks implementing key supervised machine learning algorithms using Python and scikit-learn. Each notebook demonstrates a different algorithm, covering both regression and classification tasks, with practical examples and code explanations.

## Types of Supervised Algorithms

### Regression Algorithms (predict continuous values)

- **Linear Regression** – Predicts a straight-line relationship between variables.
- **Polynomial Regression** – Captures non-linear relationships between features and the target variable.
- **Ridge/Lasso Regression** – Regularized versions of linear regression to reduce overfitting (Ridge uses L2 regularization, Lasso uses L1).
- **Support Vector Regression (SVR)** – Uses support vectors to predict continuous values, effective in high-dimensional spaces.
- **Decision Tree Regressor** – Predicts numeric outcomes using tree-based splits.
- **Random Forest Regressor** – An ensemble of decision trees that improves robustness and accuracy in regression tasks.
- **Gradient Boosted Regression (e.g., XGBoost, LightGBM, CatBoost)** – Sequential ensemble techniques that build strong predictive models by combining weak learners.

### Classification Algorithms (predict discrete classes)

- **Logistic Regression** – Used for binary or multi-class classification problems.
- **k-Nearest Neighbors (kNN)** – Classifies based on the most common class among the closest neighbors in the feature space.
- **Support Vector Machines (SVM)** – Finds optimal hyperplanes to separate classes in the feature space.
- **Decision Tree Classifier** – Uses tree-based rules to assign classes to input features.
- **Random Forest Classifier** – An ensemble of decision trees for improved classification accuracy and stability.
- **Gradient Boosted Trees (e.g., XGBoost, LightGBM, CatBoost)** – Builds strong classifiers by combining many weak tree-based learners sequentially.
- **Naïve Bayes** – Probabilistic classifier based on Bayes’ theorem, particularly effective for text classification.
- **Neural Networks** – Can classify complex and high-dimensional data; deep learning architectures enable modeling of intricate patterns.

---

These algorithms provide a foundation for solving a wide variety of supervised machine learning problems, including both regression (predicting continuous values) and classification (predicting discrete labels) tasks.

- **LinearRegression.ipynb**  
  Implementation of Linear Regression for regression tasks, using the diabetes dataset.

- **LogisticRegression.ipynb**  
  Implementation of Logistic Regression for binary classification, demonstrated with the breast cancer dataset.

- **KNN.ipynb**  
  Implementation of k-Nearest Neighbors (kNN) classifier, including hyperparameter tuning and evaluation.

- **DecisionTreeClassifier.ipynb**  
  Example of Decision Tree Classifier for classification tasks, with visualization and performance metrics.

## Requirements

- Python 3.x
- numpy
- matplotlib
- scikit-learn

Install dependencies using pip:

```bash
pip install numpy matplotlib scikit-learn
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/ranjithayandrapati/Machine-Learning-Algorithms.git
    cd Machine-Learning-Algorithms/Supervised\ Algorithms
    ```

2. Open any notebook in Jupyter or Google Colab:
    ```bash
    jupyter notebook
    ```
   Or upload to [Google Colab](https://colab.research.google.com/) directly.

3. Run the cells to see the step-by-step implementation and results.

## References

- [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

---

For an overview of all algorithms in this repository, see the main [README.md](../README.md).

