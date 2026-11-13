# Ensemble Methods

An ensemble combines several fitted models into one predictor. The key idea is
not merely to train more models, but to combine models whose errors differ.
Agreement among identical models adds little; diversity makes aggregation
useful.

## Why combine models?

Suppose classifiers $h_1,\ldots,h_M$ each predict a class for an observation
$x$. A simple voting ensemble predicts

$$
\hat{y}(x)=\operatorname{mode}\{h_1(x),\ldots,h_M(x)\}.
$$

If the learners are reasonably accurate and do not make perfectly correlated
errors, incorrect votes can cancel. This produces a useful bias-variance
tradeoff:

- **Bagging** primarily reduces variance by averaging independently trained
  learners.
- **Boosting** primarily reduces bias by fitting new learners to the current
  ensemble's mistakes.

## Parallel and sequential ensembles

In a parallel ensemble, every learner can be fitted independently. Bagging
draws a bootstrap sample for each learner, fits each model, and aggregates the
results. Random forests strengthen this idea by randomizing the features
available at each tree split.

In a sequential ensemble, learner $m$ depends on learners
$1,\ldots,m-1$. AdaBoost increases the influence of misclassified observations
before fitting the next learner. The final classifier uses a weighted vote.

## A consistent API

The ensemble classifiers follow the same interface as the other `rice_ml`
estimators:

```python
from rice_ml.supervised_ml import AdaBoostClassifier, RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
forest_predictions = forest.predict(X_test)

boost = AdaBoostClassifier(n_estimators=50, learning_rate=0.8)
boost.fit(X_train, y_train)
boost_predictions = boost.predict(X_test)
```

Both models also provide `predict_proba`, `classes_`, `n_features_in_`, and
`feature_importances_` after fitting.

## Choosing an ensemble

Random forests are a strong first choice for tabular data: they capture
nonlinear structure, require little preprocessing, and are comparatively
robust to tuning. AdaBoost is attractive when simple rules can be combined
into a strong boundary, but mislabeled points and outliers deserve extra
attention because boosting repeatedly emphasizes difficult observations.

## Review questions

1. Why is diversity among learners important?
2. Which ensemble strategy is naturally parallelizable?
3. How do bagging and boosting affect bias and variance differently?

Continue with the [random forests chapter](random_forests.md), or work through
the [ensemble methods notebook](../../notebooks/ensemble_methods/01_ensemble_methods.ipynb).
