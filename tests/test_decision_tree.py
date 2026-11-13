import numpy as np
import pytest

from rice_ml.supervised_ml import DecisionTreeClassifier


def test_decision_tree_classifier_predicts_basic_labels():
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0, 0, 1, 1]

    model = DecisionTreeClassifier(max_depth=1)
    model.fit(X, y)

    preds = model.predict([[0.5], [2.5]])
    assert preds.tolist() == [0, 1]


def test_decision_tree_classifier_supports_entropy():
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = ["low", "low", "high", "high"]

    model = DecisionTreeClassifier(criterion="entropy", max_depth=1)
    model.fit(X, y)

    assert model.predict([[2.2]]).tolist() == ["high"]


def test_decision_tree_classifier_raises_before_fit():
    model = DecisionTreeClassifier()
    with pytest.raises(ValueError, match="fitted"):
        model.predict([[1.0]])


def test_decision_tree_classifier_invalid_hyperparameters_raise():
    with pytest.raises(ValueError, match="criterion"):
        DecisionTreeClassifier(criterion="misclassification")

    with pytest.raises(ValueError, match="max_depth"):
        DecisionTreeClassifier(max_depth=0)

    with pytest.raises(ValueError, match="min_samples_split"):
        DecisionTreeClassifier(min_samples_split=1)

    with pytest.raises(ValueError, match="min_samples_leaf"):
        DecisionTreeClassifier(min_samples_leaf=0)


def test_decision_tree_classifier_rejects_bad_feature_counts():
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0, 0, 1, 1]

    model = DecisionTreeClassifier(max_depth=1)
    model.fit(X, y)

    with pytest.raises(ValueError, match="same number of features"):
        model.predict([[0.0, 1.0]])


def test_decision_tree_classifier_uses_max_depth_as_a_stopping_rule():
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0, 0, 1, 0]

    shallow = DecisionTreeClassifier(max_depth=1)
    shallow.fit(X, y)

    deep = DecisionTreeClassifier(max_depth=2)
    deep.fit(X, y)

    assert shallow.predict(X).tolist() == [0, 0, 0, 0]
    assert deep.predict(X).tolist() == y


def test_decision_tree_classifier_sets_user_friendly_attributes():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = DecisionTreeClassifier(max_depth=1)
    model.fit(X, y)

    assert model.classes_.tolist() == [0, 1]
    assert model.n_features_in_ == 1
    assert model.tree_depth_ == 1
