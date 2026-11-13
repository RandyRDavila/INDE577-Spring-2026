# Boosting

Boosting constructs a strong classifier from a sequence of weak classifiers.
Unlike a random forest, later learners are trained in response to earlier
mistakes.

## AdaBoost with decision stumps

`AdaBoostClassifier` uses decision stumps: trees with one split. Initially,
every training observation receives weight $w_i=1/n$. At round $m$, the stump
$h_m$ minimizes weighted classification error

$$
\epsilon_m=\sum_{i=1}^{n}w_i
\mathbf{1}\bigl(h_m(x_i)\ne y_i\bigr).
$$

For $K$ classes, the SAMME voting weight is

$$
\alpha_m=\eta\left[
\log\left(\frac{1-\epsilon_m}{\epsilon_m}\right)+\log(K-1)
\right],
$$

where $\eta$ is the learning rate. Lower-error stumps receive larger votes.
Misclassified observations are then upweighted:

$$
w_i\leftarrow w_i\exp\left(
\alpha_m\mathbf{1}(h_m(x_i)\ne y_i)
\right),
$$

followed by normalization so the weights sum to one.

## Fitting AdaBoost

```python
from rice_ml.supervised_ml import AdaBoostClassifier

boost = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=0.8,
    random_state=42,
)
boost.fit(X_train, y_train)

predictions = boost.predict(X_test)
probabilities = boost.predict_proba(X_test)
```

After fitting, `estimators_` contains the retained stumps,
`estimator_errors_` records their weighted errors, and
`estimator_weights_` contains their voting weights.

## Learning rate and number of rounds

`learning_rate` shrinks every stump's contribution. A smaller value commonly
needs more estimators but can produce smoother learning. A large value adapts
quickly and can focus aggressively on difficult or mislabeled observations.

The algorithm stops early when a stump classifies the weighted training set
perfectly. It also rejects a stump that is no better than random guessing under
the multiclass SAMME criterion.

## Strengths and cautions

Boosting can turn simple, interpretable rules into a flexible nonlinear
classifier. It often performs well without deep trees. However, because hard
observations gain weight, outliers and label errors can attract disproportionate
attention. Inspect validation performance as the number of rounds grows.

## Review questions

1. Why does a low-error stump receive a larger $\alpha_m$?
2. What happens to a misclassified observation before the next round?
3. Why might a smaller learning rate require more estimators?

Complete the [boosting notebook](../../notebooks/ensemble_methods/03_boosting.ipynb)
to inspect stump errors and voting weights directly.
