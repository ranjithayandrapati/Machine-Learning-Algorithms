# Machine-Learning-Algorithms

# Supervised Algorithms

This folder contains Jupyter notebooks implementing key supervised machine learning algorithms using Python and scikit-learn. Each notebook demonstrates a different algorithm, covering both regression and classification tasks, with practical examples and code explanations.

## Contents

- **LinearRegression.ipynb**  
  Implementation of Linear Regression for regression tasks, using the diabetes dataset.

- **LogisticRegression.ipynb**  
  Implementation of Logistic Regression for binary classification, demonstrated with the breast cancer dataset.

- **KNN.ipynb**  
  Implementation of k-Nearest Neighbors (kNN) classifier, including hyperparameter tuning and evaluation.

- **DecisionTreeClassifier.ipynb**  
  Example of Decision Tree Classifier for classification tasks, with visualization and performance metrics.

## Algorithms Covered

- Linear Regression
- Logistic Regression
- k-Nearest Neighbors (kNN)
- Decision Tree Classifier

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


#Types of Supervised Algorithms

 Regression Algorithms (predict continuous values)

Linear Regression – predicts a straight-line relationship between variables.

Polynomial Regression – captures non-linear relationships.

Ridge/Lasso Regression – regularized versions to reduce overfitting.

Support Vector Regression (SVR) – uses support vectors to predict continuous values.

Decision Tree Regressors – tree-based prediction of numeric outcomes.

Random Forest Regressors – ensemble of decision trees for robust regression.

Gradient Boosted Regression (XGBoost, LightGBM, CatBoost).

Classification Algorithms (predict discrete classes)

Logistic Regression – for binary or multi-class classification.

k-Nearest Neighbors (kNN) – predicts based on closest neighbors.

Support Vector Machines (SVM) – finds hyperplanes separating classes.

Decision Trees – tree-based rules to classify.

Random Forest Classifiers – ensemble of trees for better accuracy.

Gradient Boosted Trees – sequential tree-based models.

Naïve Bayes – probabilistic classifier based on Bayes’ theorem.

Neural Networks – can classify complex patterns (deep learning).
