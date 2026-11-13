"""Ensemble classifiers for ``rice_ml``.

Examples
--------
>>> from rice_ml.supervised_ml.ensemble import RandomForestClassifier
>>> model = RandomForestClassifier(n_estimators=10, random_state=0)
>>> _ = model.fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
>>> model.predict([[0.25], [2.75]]).tolist()
[0, 1]
"""

from .models import AdaBoostClassifier, RandomForestClassifier

__all__ = ["AdaBoostClassifier", "RandomForestClassifier"]
