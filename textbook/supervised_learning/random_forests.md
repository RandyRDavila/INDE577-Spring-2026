# Random Forests

A decision tree is expressive and interpretable, but small changes in the
training data can produce a very different tree. A random forest stabilizes
that learner by averaging many deliberately different trees.

## Bootstrap aggregation

For a training set containing $n$ observations, a bootstrap sample draws $n$
indices **with replacement**. Some observations appear multiple times and
others are omitted. Each tree receives its own bootstrap sample.

For classification, tree probabilities are averaged:

$$
\hat{p}(Y=k\mid x)=\frac{1}{M}\sum_{m=1}^{M}
\hat{p}_m(Y=k\mid x),
$$

and the class with the largest average probability is selected.

## Random feature selection

Bootstrap samples randomize observations, but strong predictors can still make
the trees resemble one another. A random forest therefore exposes each split
to only `max_features` candidate features. This decorrelates the trees and can
make averaging more effective.

The `rice_ml` classifier accepts:

- `"sqrt"`: $\max(1,\lfloor\sqrt{p}\rfloor)$ features per split.
- `"log2"`: $\max(1,\lfloor\log_2 p\rfloor)$ features per split.
- An integer feature count.
- A float in $(0,1]$ representing a fraction of all features.
- `None` to consider every feature.

## Fitting a forest

```python
from rice_ml.supervised_ml import RandomForestClassifier

forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=2,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
)
forest.fit(X_train, y_train)

predictions = forest.predict(X_test)
probabilities = forest.predict_proba(X_test)
```

`random_state` makes bootstrap samples and feature choices reproducible.
Increasing `n_estimators` usually stabilizes predictions but increases training
and prediction cost. Tree-size controls such as `max_depth` and
`min_samples_leaf` regulate the complexity of each ensemble member.

## Feature importance

`feature_importances_` accumulates the weighted decrease in Gini impurity
attributed to each feature and normalizes the values to sum to one. It answers
“which features did this fitted forest use to make pure child nodes?” It does
not establish causality, and correlated predictors can divide or distort
importance.

## Practical guidance

- Start with `max_features="sqrt"` and a fixed `random_state`.
- Increase `n_estimators` until validation performance stabilizes.
- Use `max_depth` or `min_samples_leaf` if individual trees overfit noise.
- Evaluate on held-out data; training accuracy is often optimistic.
- Treat feature importance as a diagnostic, not a causal conclusion.

Work through the [random forests notebook](../../notebooks/ensemble_methods/02_random_forests.ipynb),
then continue to [boosting](boosting.md).
