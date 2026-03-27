import math

import numpy as np
import pandas as pd
import pytest

from rice_ml.supervised_ml.knn import KNNClassifier, KNNRegressor, KNNRecommender


def chebyshev_distance(x, y):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    return float(np.max(np.abs(x_arr - y_arr)))


def test_knn_classifier_predicts_basic_labels():
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0, 0, 1, 1]

    model = KNNClassifier(n_neighbors=3)
    model.fit(X, y)

    preds = model.predict([[0.9], [2.1]])
    assert preds.tolist() == [0, 1]


def test_knn_classifier_supports_taxicab_distance():
    X = [[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]]
    y = ["a", "a", "b"]

    model = KNNClassifier(n_neighbors=1, distance="taxicab")
    model.fit(X, y)

    assert model.predict([[0.5, 0.5]]).tolist() == ["a"]


def test_knn_classifier_supports_custom_distance():
    X = [[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]]
    y = [0, 0, 1]

    model = KNNClassifier(n_neighbors=1, distance=chebyshev_distance)
    model.fit(X, y)

    assert model.predict([[0.2, 0.1]]).tolist() == [0]


def test_knn_classifier_raises_before_fit():
    model = KNNClassifier(n_neighbors=1)
    with pytest.raises(ValueError, match="fitted"):
        model.predict([[1.0]])


def test_knn_classifier_invalid_neighbors_raise():
    with pytest.raises(ValueError, match="positive integer"):
        KNNClassifier(n_neighbors=0)


def test_knn_regressor_predicts_mean_of_neighbors():
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.0, 1.0, 4.0, 9.0]

    model = KNNRegressor(n_neighbors=2)
    model.fit(X, y)

    pred = model.predict([[1.5]])[0]
    assert math.isclose(pred, 2.5)


def test_knn_regressor_distance_weighting():
    X = [[0.0], [2.0], [5.0]]
    y = [0.0, 4.0, 10.0]

    model = KNNRegressor(n_neighbors=2, weights="distance")
    model.fit(X, y)

    pred = model.predict([[1.0]])[0]
    assert 0.0 < pred < 4.0


def test_knn_regressor_rejects_non_numeric_targets():
    model = KNNRegressor(n_neighbors=1)
    with pytest.raises(TypeError, match="numeric"):
        model.fit([[0.0], [1.0]], ["a", "b"])


def test_knn_recommender_predict_rating_and_recommend():
    ratings = [
        [5.0, 4.0, np.nan, np.nan],
        [4.0, np.nan, 5.0, 3.0],
        [1.0, 2.0, 1.0, np.nan],
        [5.0, 5.0, 4.0, 4.0],
    ]

    model = KNNRecommender(n_neighbors=2, distance="euclidean", weights="distance")
    model.fit(ratings)

    pred = model.predict_rating(user_index=0, item_index=2)
    assert np.isfinite(pred)
    assert pred > 0.0

    recs = model.recommend(user_index=0, top_n=2)
    assert isinstance(recs, list)
    assert len(recs) >= 1
    assert all(isinstance(item_idx, int) for item_idx, _ in recs)


def test_knn_recommender_returns_dataframe():
    ratings = [
        [5.0, np.nan, np.nan],
        [4.0, 3.0, 5.0],
        [5.0, 4.0, 4.0],
    ]

    model = KNNRecommender(n_neighbors=2)
    model.fit(ratings)

    frame = model.recommend_frame(user_index=0, top_n=2)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["item_index", "predicted_rating"]


def test_knn_recommender_rejects_bad_indices():
    ratings = [
        [5.0, np.nan],
        [4.0, 3.0],
    ]

    model = KNNRecommender(n_neighbors=1)
    model.fit(ratings)

    with pytest.raises(IndexError, match="user_index"):
        model.predict_rating(user_index=5, item_index=0)

    with pytest.raises(IndexError, match="item_index"):
        model.predict_rating(user_index=0, item_index=5)


def test_knn_recommender_rejects_invalid_top_n():
    ratings = [
        [5.0, np.nan],
        [4.0, 3.0],
    ]

    model = KNNRecommender(n_neighbors=1)
    model.fit(ratings)

    with pytest.raises(ValueError, match="top_n"):
        model.recommend(user_index=0, top_n=0)
