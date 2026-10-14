"""Machine-learning logic for the visualiser: the models, how the data is prepared
for them, and how they are scored.

The models, their settings, the stratified 70/30 split (seed 42) and the metrics
come from the original Streamlit app (legacy/streamlit_app.py). What changed when
the app learned to handle any dataset is the preparation of the features:

  * number columns: missing values are filled with the median; they are
    standardised for the models that need it (SVM, logistic regression, KNN)
  * category columns: missing values become their own category, and every
    category gets its own 0/1 column (one-hot), so no order is invented
  * Naive Bayes is the exception: it models number columns with a bell curve
    and category columns with category frequencies (see MixedNaiveBayes)

All of this lives inside the pipeline, so it is learned from training rows only.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, learning_curve, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score,
    classification_report, confusion_matrix, average_precision_score,
    roc_curve, precision_recall_curve,
)

from .datasets import DataError, Problem

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
NEEDS_SCALING  = {"Support Vector Machine (SVM)", "Logistic Regression", "K-Nearest Neighbors (KNN)"}
TREE_BASED     = {"Random Forest", "Decision Tree", "Gradient Boosting"}

# The classifiers and the hyperparameters the UI may change — the same widgets,
# ranges and defaults the Streamlit sidebar had.
CLASSIFIERS = [
    {
        "name": "Support Vector Machine (SVM)", "short": "SVM",
        "params": [
            {"key": "C", "label": "C (regularisation)", "type": "float",
             "min": 0.01, "max": 20.0, "step": 0.05, "default": 1.0,
             "help": "Higher C means less regularisation: the model fits the training data more tightly."},
            {"key": "kernel", "label": "Kernel", "type": "choice",
             "options": ["rbf", "linear", "poly", "sigmoid"], "default": "rbf",
             "help": "The kernel transforms the feature space."},
            {"key": "gamma", "label": "Gamma", "type": "choice",
             "options": ["scale", "auto"], "default": "scale",
             "help": "Kernel coefficient (scale = 1 / (n_features · Var(X)))."},
        ],
    },
    {
        "name": "Logistic Regression", "short": "Logistic Reg.",
        "params": [
            {"key": "C", "label": "C (regularisation)", "type": "float",
             "min": 0.01, "max": 20.0, "step": 0.05, "default": 1.0,
             "help": "Higher C means less regularisation."},
            {"key": "max_iter", "label": "Max iterations", "type": "int",
             "min": 100, "max": 3000, "step": 100, "default": 1000},
            {"key": "solver", "label": "Solver", "type": "choice",
             "options": ["lbfgs", "saga", "liblinear"], "default": "lbfgs"},
        ],
    },
    {
        "name": "Random Forest", "short": "Random Forest",
        "params": [
            {"key": "n_estimators", "label": "Number of trees", "type": "int",
             "min": 10, "max": 500, "step": 10, "default": 100,
             "help": "More trees usually means higher accuracy and slower training."},
            {"key": "max_depth", "label": "Max depth", "type": "int",
             "min": 1, "max": 40, "step": 1, "default": 15},
            {"key": "min_samples_split", "label": "Min samples to split", "type": "int",
             "min": 2, "max": 20, "step": 1, "default": 2},
            {"key": "max_features", "label": "Max features per split", "type": "choice",
             "options": ["sqrt", "log2"], "default": "sqrt",
             "help": "sqrt is the standard choice for classification."},
        ],
    },
    {
        "name": "K-Nearest Neighbors (KNN)", "short": "KNN",
        "params": [
            {"key": "n_neighbors", "label": "K (neighbours)", "type": "int",
             "min": 1, "max": 30, "step": 1, "default": 5},
            {"key": "weights", "label": "Weight function", "type": "choice",
             "options": ["uniform", "distance"], "default": "uniform"},
            {"key": "metric", "label": "Distance metric", "type": "choice",
             "options": ["minkowski", "euclidean", "manhattan"], "default": "minkowski"},
            {"key": "algorithm", "label": "Algorithm", "type": "choice",
             "options": ["auto", "ball_tree", "kd_tree"], "default": "auto"},
        ],
    },
    {
        "name": "Decision Tree", "short": "Decision Tree",
        "params": [
            {"key": "max_depth", "label": "Max depth", "type": "int",
             "min": 1, "max": 30, "step": 1, "default": 10,
             "help": "A deeper tree risks overfitting."},
            {"key": "criterion", "label": "Split criterion", "type": "choice",
             "options": ["gini", "entropy", "log_loss"], "default": "gini"},
            {"key": "min_samples_split", "label": "Min samples to split", "type": "int",
             "min": 2, "max": 20, "step": 1, "default": 2},
            {"key": "min_samples_leaf", "label": "Min samples per leaf", "type": "int",
             "min": 1, "max": 20, "step": 1, "default": 1},
            {"key": "ccp_alpha", "label": "CCP alpha (pruning)", "type": "float",
             "min": 0.0, "max": 0.05, "step": 0.001, "default": 0.0,
             "help": "A larger alpha prunes more, which reduces overfitting."},
        ],
    },
    {
        "name": "Gradient Boosting", "short": "Gradient Boosting",
        "params": [
            {"key": "n_estimators", "label": "Boosting stages", "type": "int",
             "min": 50, "max": 500, "step": 10, "default": 100},
            {"key": "learning_rate", "label": "Learning rate", "type": "float",
             "min": 0.005, "max": 0.5, "step": 0.005, "default": 0.1,
             "help": "A lower rate with more stages often gives the best results."},
            {"key": "max_depth", "label": "Max tree depth", "type": "int",
             "min": 1, "max": 10, "step": 1, "default": 3},
            {"key": "subsample", "label": "Subsample fraction", "type": "float",
             "min": 0.4, "max": 1.0, "step": 0.05, "default": 1.0,
             "help": "Below 1.0 each stage sees a random part of the data, which often generalises better."},
        ],
    },
    {
        "name": "Naive Bayes", "short": "Naive Bayes",
        "params": [
            {"key": "var_smoothing_exp", "label": "Var smoothing (1e-x)", "type": "int",
             "min": 1, "max": 15, "step": 1, "default": 9, "applies_to": "number",
             "help": "For number columns. Smoothing is 10 to the power of minus x, so a larger x means less smoothing."},
            {"key": "alpha", "label": "Category smoothing", "type": "float",
             "min": 0.01, "max": 5.0, "step": 0.05, "default": 1.0, "applies_to": "category",
             "help": "For category columns. A higher value trusts rare categories less."},
        ],
    },
]
_SPEC = {c["name"]: c for c in CLASSIFIERS}


def resolve_params(clf_name, params):
    """Fill in defaults and check every value against the allowed range."""
    if clf_name not in _SPEC:
        raise ValueError(f"Unknown classifier: {clf_name!r}")
    params = params or {}
    out = {}
    for spec in _SPEC[clf_name]["params"]:
        key = spec["key"]
        value = params.get(key, spec["default"])
        if spec["type"] == "choice":
            if value not in spec["options"]:
                raise ValueError(f"{key} must be one of {spec['options']}")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number")
            if not spec["min"] <= value <= spec["max"]:
                raise ValueError(f"{key} must be between {spec['min']} and {spec['max']}")
            value = int(round(value)) if spec["type"] == "int" else float(value)
        out[key] = value
    return out


def make_classifier(clf_name, p):
    if clf_name == "Support Vector Machine (SVM)":
        return SVC(C=p["C"], kernel=p["kernel"], gamma=p["gamma"], probability=True, random_state=42)

    elif clf_name == "Logistic Regression":
        return LogisticRegression(C=p["C"], max_iter=p["max_iter"], solver=p["solver"], random_state=42)

    elif clf_name == "Random Forest":
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                      min_samples_split=p["min_samples_split"],
                                      max_features=p["max_features"],
                                      random_state=42, n_jobs=-1)

    elif clf_name == "K-Nearest Neighbors (KNN)":
        return KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"],
                                    metric=p["metric"], algorithm=p["algorithm"])

    elif clf_name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=p["max_depth"], criterion=p["criterion"],
                                      min_samples_split=p["min_samples_split"],
                                      min_samples_leaf=p["min_samples_leaf"],
                                      ccp_alpha=p["ccp_alpha"], random_state=42)

    elif clf_name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                                          max_depth=p["max_depth"], subsample=p["subsample"],
                                          random_state=42)

    elif clf_name == "Naive Bayes":
        return MixedNaiveBayes(var_smoothing=10 ** (-p["var_smoothing_exp"]), alpha=p["alpha"])

    raise ValueError(f"Unknown classifier: {clf_name!r}")


class MixedNaiveBayes(ClassifierMixin, BaseEstimator):
    """Naive Bayes for tables that mix numbers and categories.

    Number columns get a bell curve per class (Gaussian NB); category columns get a
    table of category frequencies per class (Categorical NB). Being "naive", the
    model treats every column as independent, so the two parts simply add up:

        log P(class | row) = log P(class | numbers) + log P(class | categories) - log P(class)

    It expects the number columns first, then the category columns as whole-number
    codes where -1 means "a category not seen in training" (see build_pipeline).
    """

    def __init__(self, n_numeric=0, var_smoothing=1e-9, alpha=1.0):
        self.n_numeric = n_numeric
        self.var_smoothing = var_smoothing
        self.alpha = alpha

    def _split(self, X):
        X = np.asarray(X, dtype=float)
        # shift the codes by one so that "unseen" (-1) becomes category 0
        return X[:, :self.n_numeric], X[:, self.n_numeric:].astype(int) + 1

    def fit(self, X, y):
        numbers, codes = self._split(X)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_log_prior_ = np.log(counts / counts.sum())
        self.gaussian_ = None
        self.categorical_ = None
        if numbers.shape[1]:
            self.gaussian_ = GaussianNB(var_smoothing=self.var_smoothing).fit(numbers, y)
        if codes.shape[1]:
            self.categorical_ = CategoricalNB(
                alpha=self.alpha, min_categories=codes.max(axis=0) + 1).fit(codes, y)
        return self

    def predict_log_proba(self, X):
        numbers, codes = self._split(X)
        parts = []
        if self.gaussian_ is not None:
            parts.append(self.gaussian_.predict_log_proba(numbers))
        if self.categorical_ is not None:
            parts.append(self.categorical_.predict_log_proba(codes))
        joint = sum(parts) - (len(parts) - 1) * self.class_log_prior_
        return joint - np.logaddexp.reduce(joint, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def build_pipeline(clf_name, clf, problem: Problem):
    """Data preparation followed by the classifier, as one object that is fitted on training rows."""
    numbers = [("fill", SimpleImputer(strategy="median"))]
    if clf_name in NEEDS_SCALING:
        numbers.append(("scale", StandardScaler()))
    categories = [("fill", SimpleImputer(strategy="constant", fill_value="missing"))]
    if isinstance(clf, MixedNaiveBayes):
        # Naive Bayes counts categories directly, so they stay one column each (as codes)
        clf.set_params(n_numeric=len(problem.numeric))
        categories.append(("codes", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
    else:
        categories.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
    columns = []
    if problem.numeric:
        columns.append(("num", Pipeline(numbers), problem.numeric))
    if problem.categorical:
        columns.append(("cat", Pipeline(categories), problem.categorical))
    prepare = ColumnTransformer(columns, remainder="drop", sparse_threshold=0)
    return Pipeline([("prep", prepare), ("clf", clf)])


# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def _scores(pipe, X):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)

def compute_metrics(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    m = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score":  f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "MCC":       (matthews_corrcoef(y_test, y_pred) + 1) / 2,   # normalise to 0-1
        "Cohen κ":   (cohen_kappa_score(y_test, y_pred) + 1) / 2,
    }
    for k in ("ROC-AUC", "Avg Precision"):
        m[k] = None
    try:
        proba = _scores(pipe, X_test)
        m["ROC-AUC"]      = roc_auc_score(y_test, proba)
        m["Avg Precision"] = average_precision_score(y_test, proba)
    except Exception:
        pass
    return m, y_pred


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAIN & EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

def fit_pipeline(problem: Problem, clf_name, params):
    pipe = build_pipeline(clf_name, make_classifier(clf_name, params), problem)
    pipe.fit(problem.X_train, problem.y_train)
    return pipe


def _thin(*arrays, limit=400):
    """Keep at most `limit` evenly spaced points of a curve (first and last always kept)."""
    n = len(arrays[0])
    if n <= limit:
        return [a.tolist() for a in arrays]
    idx = np.unique(np.linspace(0, n - 1, limit).round().astype(int))
    return [a[idx].tolist() for a in arrays]


def evaluate(problem: Problem, pipe):
    X_test, y_test = problem.X_test, problem.y_test
    metrics, y_pred = compute_metrics(pipe, X_test, y_test)
    out = {
        "class_names": problem.class_names,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "report": classification_report(y_pred=y_pred, y_true=y_test, labels=[0, 1],
                                        target_names=["negative", "positive"],
                                        output_dict=True, zero_division=0),
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "roc": None,
        "pr": None,
    }
    try:
        proba = _scores(pipe, X_test)
        fpr, tpr, _ = roc_curve(y_test, proba)
        precision, recall, _ = precision_recall_curve(y_test, proba)
        out["roc"] = dict(zip(("fpr", "tpr"), _thin(fpr, tpr)))
        # precision_recall_curve runs from recall 1 down to 0; flip it to read left to right
        out["pr"] = dict(zip(("recall", "precision"), _thin(recall[::-1], precision[::-1])))
    except Exception:
        pass
    return out


def _owners(pipe):
    """For every column the classifier sees, the original feature it was made from."""
    owners = []
    for name, transformer, columns in pipe.named_steps["prep"].transformers_:
        if name == "num":
            owners += list(columns)
        elif name == "cat":
            encoder = transformer.named_steps["onehot"]
            for column, categories in zip(columns, encoder.categories_):
                owners += [column] * len(categories)
    return owners


def feature_importance(problem: Problem, pipe, clf_name, n_top=15):
    features = problem.features
    if clf_name in TREE_BASED:
        # a category feature is spread over several 0/1 columns: add their shares back together
        shares = pd.Series(pipe.named_steps["clf"].feature_importances_, index=_owners(pipe))
        importances = shares.groupby(level=0).sum().reindex(features).to_numpy()
        idx = np.argsort(importances)[-n_top:][::-1]
        return {
            "method": "impurity",
            "items": [{"feature": features[i], "importance": float(importances[i]), "std": None}
                      for i in idx],
        }
    result = permutation_importance(pipe, problem.X_test, problem.y_test,
                                    n_repeats=15, random_state=42, n_jobs=-1)
    idx = np.argsort(result.importances_mean)[-n_top:][::-1]
    return {
        "method": "permutation",
        "items": [{"feature": features[i],
                   "importance": float(result.importances_mean[i]),
                   "std": float(result.importances_std[i])}
                  for i in idx],
    }


def learning_curve_data(problem: Problem, pipe):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        sizes, tr_scores, cv_scores = learning_curve(
            pipe, problem.X_train, problem.y_train, cv=cv,
            train_sizes=np.linspace(.1, 1.0, 9),
            scoring="accuracy", n_jobs=-1, error_score="raise",
        )
    except ValueError as error:
        if "class" in str(error):
            raise DataError("There are too few rows for a learning curve: the smallest slices "
                            "of the training set do not contain both outcomes.")
        raise DataError(f"The learning curve could not be computed ({error}).")
    tr_m, tr_s = tr_scores.mean(1), tr_scores.std(1)
    cv_m, cv_s = cv_scores.mean(1), cv_scores.std(1)
    return {
        "sizes": sizes.tolist(),
        "train_mean": tr_m.tolist(), "train_std": tr_s.tolist(),
        "cv_mean": cv_m.tolist(), "cv_std": cv_s.tolist(),
        "gap": float(tr_m[-1] - cv_m[-1]),
    }


def cross_validation(problem: Problem, pipe):
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, problem.X, problem.y,
                             cv=cv, scoring="accuracy", n_jobs=-1)
    return {
        "scores": scores.tolist(),
        "mean": float(scores.mean()), "std": float(scores.std()),
        "min": float(scores.min()), "max": float(scores.max()),
    }


def compare_all_models(problem: Problem):
    candidates = {
        "SVM":               ("Support Vector Machine (SVM)",
                              SVC(kernel="rbf", probability=True, random_state=42)),
        "Logistic Reg.":     ("Logistic Regression",
                              LogisticRegression(max_iter=1000, random_state=42)),
        "Random Forest":     ("Random Forest",
                              RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        "KNN":               ("K-Nearest Neighbors (KNN)",
                              KNeighborsClassifier(n_neighbors=5)),
        "Decision Tree":     ("Decision Tree",
                              DecisionTreeClassifier(max_depth=12, random_state=42)),
        "Gradient Boosting": ("Gradient Boosting",
                              GradientBoostingClassifier(n_estimators=100, random_state=42)),
        "Naive Bayes":       ("Naive Bayes",
                              MixedNaiveBayes()),
    }
    rows = []
    for name, (clf_name, clf) in candidates.items():
        pipe = build_pipeline(clf_name, clf, problem)
        pipe.fit(problem.X_train, problem.y_train)
        m, _ = compute_metrics(pipe, problem.X_test, problem.y_test)
        m["Model"] = name
        rows.append(m)
    return pd.DataFrame(rows).set_index("Model")


# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICT ONE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

def predict_one(problem: Problem, pipe, values: dict):
    """Classify a single example given as {feature: value}. Missing features are allowed."""
    row = {}
    for name in problem.numeric:
        value = values.get(name)
        try:
            row[name] = np.nan if value in (None, "") else float(value)
        except (TypeError, ValueError):
            raise DataError(f"'{name}' must be a number.")
    for name in problem.categorical:
        value = values.get(name)
        row[name] = np.nan if value in (None, "") else str(value)
    frame = pd.DataFrame([row], columns=problem.features).astype(
        {name: float for name in problem.numeric} | {name: object for name in problem.categorical})

    label = int(pipe.predict(frame)[0])
    out = {"label": problem.class_names[label], "is_positive": bool(label), "probability": None}
    if hasattr(pipe, "predict_proba"):
        out["probability"] = float(pipe.predict_proba(frame)[0, 1])
    return out
