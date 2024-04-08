# pyallffs
PYALLFFS: An open-source library for all filter feature selection methods

_____________________
# Feature Selection

Feature selection is a technique used in machine learning and data mining to improve model performance, reduce irrelevant information, and decrease computational costs. It aims to identify the most important or influential variables among those present in the dataset. During this process, unnecessary or low-impact features are removed, or only those that contribute the most to model performance are selected. Feature selection reduces data dimensionality, thereby enhancing the model's generalization ability, reducing the risk of overfitting, and making the model simpler and more interpretable.

___________________
# Filter Methods

Filter method is one of the feature selection techniques in machine learning, which involves evaluating each feature independently of the machine learning algorithm. It ranks the features based on certain criteria, such as correlation, statistical tests, or information gain, and selects the top-ranked features for model training. Unlike wrapper and embedded methods, filter methods are computationally less expensive and less prone to overfitting, making them suitable for high-dimensional datasets. However, they may overlook interactions between features. Overall, filter methods serve as an initial step in feature selection, providing insights into the relevance of individual features to the target variable.

________________
# All Filter Methods Used in This Package:

* Fisher Score
* T-Score
* Welch's t-statistic
* Chi-Squared
* Information Gain
* Gain Ratio
* Symmetric Uncertainty Coefficient
* Relief Score
* Relief-F
* mRMR
* Absolute Pearson Correlation Coefficients
* Maximum Likelihood Feature Selection
* Least Squares Feature Selection
* Laplacian Feature Selection Score
* Mutual Information
* Euclidean Distance
* Cramer's V test
* Markov Blanket Filter
* Kruskal-Wallis test

____________
# Example of Usage:

```python
from pyallffs import AbsolutePearsonCorrelationCalculator
import pandas as pd

df = pd.read_csv("dataset.csv")
exmp_class = AbsolutePearsonCorrelationCalculator(df, drop_labels=["target_variable"], target="target_variable")

scores, ranked_features = exmp_class.calculate_absolute_pearson_correlation(plot_importance=True)
```


