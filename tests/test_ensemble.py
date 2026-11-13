import numpy as np
import pytest

from rice_ml.supervised_ml import AdaBoostClassifier, RandomForestClassifier

X = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [3.0, 3.0],
        [3.0, 4.0],
        [4.0, 3.0],
        [4.0, 4.0],
    ]
)
y = np.array(["low"] * 4 + ["high"] * 4)


def test_random_forest_predicts_labels_and_probabilities():
    model = RandomForestClassifier(n_estimators=25, max_depth=3, random_state=4)
    model.fit(X, y)

    assert model.predict([[0.2, 0.3], [3.8, 3.7]]).tolist() == ["low", "high"]
    probabilities = model.predict_proba([[0.2, 0.3], [3.8, 3.7]])
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_random_forest_is_reproducible():
    first = RandomForestClassifier(n_estimators=15, random_state=12).fit(X, y)
    second = RandomForestClassifier(n_estimators=15, random_state=12).fit(X, y)

    assert np.allclose(first.predict_proba(X), second.predict_proba(X))
    assert np.isclose(first.feature_importances_.sum(), 1.0)


def test_random_forest_resolves_max_features():
    model = RandomForestClassifier(n_estimators=3, max_features=0.5, random_state=0)
    model.fit(X, y)
    assert model.max_features_ == 1

    with pytest.raises(ValueError, match="max_features"):
        RandomForestClassifier(max_features=3).fit(X, y)


def test_adaboost_predicts_binary_labels_and_probabilities():
    model = AdaBoostClassifier(n_estimators=20, learning_rate=0.8)
    model.fit(X, y)

    assert model.predict([[0.2, 0.3], [3.8, 3.7]]).tolist() == ["low", "high"]
    probabilities = model.predict_proba([[0.2, 0.3], [3.8, 3.7]])
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert len(model.estimators_) >= 1


def test_adaboost_supports_multiclass_targets():
    multiclass_X = [[0.0], [0.5], [2.0], [2.5], [4.0], [4.5]]
    multiclass_y = ["a", "a", "b", "b", "c", "c"]
    model = AdaBoostClassifier(n_estimators=10).fit(multiclass_X, multiclass_y)

    assert model.predict([[0.25], [2.25], [4.25]]).tolist() == ["a", "b", "c"]


@pytest.mark.parametrize(
    ("model_factory", "message"),
    [
        (lambda: RandomForestClassifier(n_estimators=0), "n_estimators"),
        (lambda: RandomForestClassifier(max_depth=0), "max_depth"),
        (lambda: AdaBoostClassifier(n_estimators=0), "n_estimators"),
        (lambda: AdaBoostClassifier(learning_rate=0), "learning_rate"),
    ],
)
def test_ensemble_hyperparameter_validation(model_factory, message):
    with pytest.raises(ValueError, match=message):
        model_factory()


@pytest.mark.parametrize(
    "model",
    [RandomForestClassifier(n_estimators=3), AdaBoostClassifier(n_estimators=3)],
)
def test_ensemble_models_raise_before_fit(model):
    with pytest.raises(ValueError, match="fitted"):
        model.predict([[1.0, 1.0]])
