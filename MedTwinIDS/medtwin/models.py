"""Model factory and shared data helpers.

Three supervised detectors are compared:
  * Random Forest       - bagged decision trees (robust, interpretable baseline)
  * XGBoost             - gradient-boosted trees (strong on tabular flow data)
  * MLP (deep network)  - three-layer feed-forward neural network
An Isolation Forest trained on normal traffic only is used as an unsupervised
safety net for previously unseen (zero-day) behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import Booster, XGBClassifier

from . import config as C

MODEL_NAMES = ["Random Forest", "XGBoost", "MLP"]
CLASS_INDEX = {c: i for i, c in enumerate(C.CLASSES)}


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


class PortableXGBClassifier(XGBClassifier):
    """XGBClassifier that pickles its booster in XGBoost's portable UBJSON model format.

    The default pickle holds the booster's memory snapshot, whose random-number-generator state is
    written in a platform-specific text format, so a model saved on Linux will not load on Windows
    (and vice versa). The model format keeps everything prediction needs and is the same everywhere.
    """

    def __getstate__(self):
        state = dict(super().__getstate__())  # copy: the default state is the live __dict__
        if isinstance(state.get("_Booster"), Booster):
            state["_Booster"] = bytes(state["_Booster"].save_raw("ubj"))
        return state

    def __setstate__(self, state):
        if isinstance(state.get("_Booster"), (bytes, bytearray)):
            state["_Booster"] = Booster({"n_jobs": state.get("n_jobs")}, model_file=bytearray(state["_Booster"]))
        super().__setstate__(state)

    def __copy__(self):
        # get_params() copies the estimator: keep that a cheap shallow copy, not a booster round trip
        cp = self.__class__.__new__(self.__class__)
        cp.__dict__.update(self.__dict__)
        return cp


def make_model(name: str, seed: int = 42, n_jobs: int = -1, fast: bool = False, **params):
    if name == "Random Forest":
        p = dict(n_estimators=150 if fast else 300, max_depth=None, min_samples_leaf=1, max_features="sqrt",
                 class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed)
        p.update(params)
        return RandomForestClassifier(**p)
    if name == "XGBoost":
        p = dict(n_estimators=150 if fast else 300, max_depth=8, learning_rate=0.1, subsample=0.9,
                 colsample_bytree=0.8, min_child_weight=1, tree_method="hist", n_jobs=n_jobs,
                 random_state=seed, eval_metric="mlogloss")
        p.update(params)
        return PortableXGBClassifier(**p)
    if name == "MLP":
        p = dict(hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam", alpha=1e-4,
                 batch_size=256, learning_rate_init=1e-3, max_iter=60 if fast else 200, early_stopping=True,
                 n_iter_no_change=10, validation_fraction=0.1, random_state=seed)
        p.update(params)
        return Pipeline([("log", FunctionTransformer(signed_log)), ("scale", StandardScaler()),
                         ("mlp", MLPClassifier(**p))])
    raise ValueError(name)


def make_isolation_forest(seed: int = 42, contamination: float = 0.01):
    return Pipeline([("log", FunctionTransformer(signed_log)),
                     ("iforest", IsolationForest(n_estimators=300, max_samples=4096, contamination=contamination,
                                                 random_state=seed, n_jobs=-1))])


def oversample(X: np.ndarray, y: np.ndarray, min_per_class: int, seed: int = 42):
    """Random oversampling so that every class has at least ``min_per_class`` rows (used for the MLP)."""
    rng = np.random.default_rng(seed)
    idx = [np.arange(len(y))]
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        if 0 < len(rows) < min_per_class:
            idx.append(rng.choice(rows, min_per_class - len(rows), replace=True))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


class RemappedXGB:
    """Lets XGBoost train on a subset of classes (it requires labels 0..k-1)."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        lut = {c: i for i, c in enumerate(self.classes_)}
        self.model.fit(X, np.vectorize(lut.get)(y), sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def fit_model(name: str, model, X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Fit with the imbalance strategy appropriate to each model family."""
    if name == "XGBoost":
        w = compute_sample_weight("balanced", y)
        if not np.array_equal(np.unique(y), np.arange(len(np.unique(y)))):
            model = RemappedXGB(model)
        model.fit(X, y, sample_weight=np.sqrt(w))
    elif name == "MLP":
        Xo, yo = oversample(X, y, min_per_class=max(2000, int(0.05 * len(y))), seed=seed)
        model.fit(Xo, yo)
    else:
        model.fit(X, y)
    return model


def encode(series: pd.Series) -> np.ndarray:
    return series.map(CLASS_INDEX).to_numpy()


def xy(df: pd.DataFrame, features=None):
    features = features or C.ALL_FEATURES
    return df[features].to_numpy(dtype=np.float32), encode(df["attack"])


def full_proba(model, X, n_classes=len(C.CLASSES)):
    """predict_proba padded to the full class list (a model may be trained without some classes)."""
    p = model.predict_proba(X)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "steps"):
        classes = model.steps[-1][1].classes_
    out = np.zeros((len(X), n_classes), dtype=float)
    out[:, np.asarray(classes, dtype=int)] = p
    return out
