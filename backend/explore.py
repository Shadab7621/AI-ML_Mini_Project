"""Everything the Dataset page shows about a problem: sizes, class balance, how the
columns relate to each other and to the target, and a preview of the raw rows."""
import numpy as np
import pandas as pd

from .datasets import Problem

HEATMAP_COLUMNS = 30      # the association heatmap shows at most this many features
MISSING = "(missing)"     # a missing category counts as a value of its own, as it does for the models
TOP_FEATURES = 6
MAX_BARS = 12             # categories (or bins) per distribution chart
PREVIEW_ROWS = 20


# ── how strongly two columns are related, always between 0 and 1 ─────────────
#   number  vs number   : absolute Pearson correlation
#   category vs category: Cramér's V
#   number  vs category : correlation ratio (eta)
# Unlike a correlation of letter codes, these do not depend on how categories are ordered.

def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b).to_numpy(dtype=float)
    total = table.sum()
    if total == 0 or min(table.shape) < 2:
        return np.nan
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / (total * (min(table.shape) - 1))))


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    known = categories.notna() & values.notna()
    categories, values = categories[known], values[known]
    if len(values) < 2 or categories.nunique() < 2:
        return np.nan
    total = ((values - values.mean()) ** 2).sum()
    if total == 0:
        return np.nan
    groups = values.groupby(categories)
    between = (groups.size() * (groups.mean() - values.mean()) ** 2).sum()
    return float(np.sqrt(between / total))


def _association(a: pd.Series, a_numeric: bool, b: pd.Series, b_numeric: bool) -> float:
    if a_numeric and b_numeric:
        value = a.corr(b)
        return float(abs(value)) if pd.notna(value) else np.nan
    if a_numeric:
        return _correlation_ratio(b, a)
    if b_numeric:
        return _correlation_ratio(a, b)
    return _cramers_v(a, b)


# ── how one feature splits between the two outcomes ──────────────────────────

def _short(number: float) -> str:
    return f"{number:.3g}"


def _values(problem: Problem, feature: str) -> pd.Series:
    column = problem.X[feature]
    return column if feature in problem.numeric else column.fillna(MISSING)


def _distribution(problem: Problem, feature: str):
    column = _values(problem, feature)
    outcome = problem.y
    if feature in problem.numeric:
        known = column.notna()
        low, high = float(column[known].min()), float(column[known].max())
        edges = np.linspace(low, high, 11)
        bins = np.clip(np.digitize(column[known], edges[1:-1]), 0, 9)
        rows = []
        for i in range(10):
            inside = bins == i
            rows.append(dict(
                value=f"{_short(edges[i])} to {_short(edges[i + 1])}",
                negative=int(((outcome[known] == 0) & inside).sum()),
                positive=int(((outcome[known] == 1) & inside).sum()),
            ))
        return rows

    table = pd.crosstab(column, outcome).reindex(columns=[0, 1], fill_value=0)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False, kind="stable").index]
    rows = [dict(value=str(value), negative=int(row[0]), positive=int(row[1]))
            for value, row in table.head(MAX_BARS - 1 if len(table) > MAX_BARS else MAX_BARS).iterrows()]
    if len(table) > MAX_BARS:
        rest = table.iloc[MAX_BARS - 1:].sum()
        rows.append(dict(value="other", negative=int(rest[0]), positive=int(rest[1])))
    return rows


# ── the summary ──────────────────────────────────────────────────────────────

def summarise(problem: Problem) -> dict:
    features = problem.features
    numeric = set(problem.numeric)
    outcome = problem.y.map({0: "no", 1: "yes"})

    # every feature against the target, strongest first
    strength = {name: _association(_values(problem, name), name in numeric, outcome, False) for name in features}
    ranked = sorted(features, key=lambda name: -(strength[name] if pd.notna(strength[name]) else -1))

    shown = [name for name in features if name in set(ranked[:HEATMAP_COLUMNS])]
    labels = [problem.target] + shown
    series = {problem.target: (outcome, False)}
    series.update({name: (_values(problem, name), name in numeric) for name in shown})
    matrix = [[None] * len(labels) for _ in labels]
    for i, first in enumerate(labels):
        for j in range(i):
            second = labels[j]
            value = (strength[first] if second == problem.target
                     else _association(*series[first], *series[second]))
            matrix[i][j] = matrix[j][i] = None if pd.isna(value) else round(float(value), 4)

    top = [name for name in ranked[:TOP_FEATURES] if pd.notna(strength[name])]

    # what the form for trying one example needs to know about each feature
    fields = []
    for name in features:
        column = problem.X[name]
        if name in numeric:
            fields.append(dict(name=name, kind="number", min=float(column.min()), max=float(column.max()),
                               default=float(column.median()),
                               whole=bool(np.all(np.mod(column.dropna().to_numpy(), 1) == 0))))
        else:
            counts = column.value_counts()
            fields.append(dict(name=name, kind="category", options=[str(v) for v in counts.index],
                               default=str(counts.index[0])))

    preview = problem.frame.head(PREVIEW_ROWS)
    y = problem.y
    return dict(
        dataset=problem.dataset,
        target=problem.target,
        positive=problem.positive,
        class_names=problem.class_names,
        targets=problem.targets,
        n_rows=problem.n_rows,
        n_samples=len(problem.X),
        n_features=len(features),
        n_numeric=len(problem.numeric),
        n_categorical=len(problem.categorical),
        n_train=len(problem.X_train),
        n_test=len(problem.X_test),
        class_counts=[dict(label=problem.class_names[0], count=int((y == 0).sum())),
                      dict(label=problem.class_names[1], count=int((y == 1).sum()))],
        columns=problem.columns,
        fields=fields,
        association=dict(labels=labels, matrix=matrix, hidden=len(features) - len(shown)),
        top_features=[dict(feature=name, strength=float(strength[name]),
                           kind="number" if name in numeric else "category",
                           values=_distribution(problem, name)) for name in top],
        preview=dict(columns=preview.columns.tolist(),
                     rows=[[None if pd.isna(v) else v for v in row]
                           for row in as_rows(preview)]),
    )


def as_rows(frame: pd.DataFrame):
    """Rows of a DataFrame as plain lists (cells keep their original type)."""
    return frame.astype(object).to_numpy().tolist()


def example_row(problem: Problem, index: int) -> dict:
    """One row of the test set, as the values the prediction form works with."""
    row = problem.X_test.iloc[index % len(problem.X_test)]
    truth = int(problem.y_test.iloc[index % len(problem.y_test)])
    values = {name: (None if pd.isna(value) else (float(value) if name in problem.numeric else str(value)))
              for name, value in row.items()}
    return dict(values=values, truth=problem.class_names[truth])

