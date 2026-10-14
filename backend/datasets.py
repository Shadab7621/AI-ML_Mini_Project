"""Datasets the app can work on: the bundled samples and CSV files the user uploads.

A *problem* is a dataset plus the column to predict and the outcome to detect.
`prepare()` turns that into feature columns, a 0/1 target and the train/test split.
"""
import io
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "datasets"
UPLOAD_DIR = SAMPLE_DIR / "uploads"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 200
MIN_ROWS = 50
MIN_PER_CLASS = 10        # rows needed of each outcome (5-fold cross-validation needs at least 5)
MAX_CLASSES = 20          # more distinct values than this is not a class column
MAX_CATEGORIES = 50       # a text column with more distinct values than this is not used
CODE_VALUES = 10          # a whole-number column with at most this many distinct values is a category

TEST_SIZE = 0.3
SEED = 42


class DataError(ValueError):
    """Something about the data that the user can fix. The message is shown as it is."""


# The samples that ship with the app. `positive` is the outcome the app tries to
# detect; `labels` gives the raw class values readable names; `positive_is` says
# whether that outcome is the bad one (drawn red) or the good one (drawn green).
SAMPLES = OrderedDict(
    mushrooms=dict(
        name="Mushrooms", file=ROOT / "Mushrooms.csv", target="type", positive="p",
        labels={"e": "edible", "p": "poisonous"}, positive_is="bad", unit="mushrooms",
        tagline="Mushrooms: edible or poisonous?",
        description="UCI Mushroom dataset: each row is one mushroom, described by one-letter codes, labelled edible or poisonous.",
        miss_note="This is the dangerous mistake.",
    ),
    titanic=dict(
        name="Titanic passengers", file=SAMPLE_DIR / "titanic.csv", target="Survived", positive="1",
        labels={"0": "died", "1": "survived"}, positive_is="good", unit="passengers",
        tagline="Titanic: who survived?",
        description="Passengers of the Titanic: age, sex, ticket class and fare, and whether each one survived.",
    ),
    heart=dict(
        name="Heart disease", file=SAMPLE_DIR / "heart_disease.csv", target="heart_disease", positive="yes",
        labels={"no": "healthy", "yes": "heart disease"}, positive_is="bad", unit="patients",
        tagline="Patients: heart disease or not?",
        description="UCI Heart Disease dataset (Cleveland): test results of 303 patients and whether heart disease was found.",
        miss_note="A missed patient is the costly mistake.",
    ),
    breast_cancer=dict(
        name="Breast cancer", file=SAMPLE_DIR / "breast_cancer.csv", target="diagnosis", positive="malignant",
        labels={}, positive_is="bad", unit="tumours",
        tagline="Tumours: malignant or benign?",
        description="Breast Cancer Wisconsin dataset: 30 measurements of each tumour and its diagnosis.",
        miss_note="A missed malignant tumour is the costly mistake.",
    ),
    telco=dict(
        name="Telecom customer churn", file=SAMPLE_DIR / "telco_churn.csv", target="Churn", positive="Yes",
        labels={"No": "stayed", "Yes": "left"}, positive_is="bad", unit="customers",
        tagline="Customers: who leaves?",
        description="Customers of a telecom company: their contract, services and charges, and whether they left.",
    ),
)

_lock = threading.Lock()
_frames: dict = {}                       # dataset id -> (file version, DataFrame)
_problems: "OrderedDict[str, Problem]" = OrderedDict()


# ── finding and reading files ────────────────────────────────────────────────

def _slug(filename: str) -> str:
    stem = Path(filename).stem.lower()
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-")[:60] or "data"


def _locate(dataset_id: str):
    """Return (path, description of the dataset) or raise if it does not exist."""
    if dataset_id in SAMPLES:
        sample = SAMPLES[dataset_id]
        info = {key: sample[key] for key in ("name", "description", "unit", "tagline", "positive_is")}
        info.update(id=dataset_id, source="sample", miss_note=sample.get("miss_note"))
        return sample["file"], info
    if dataset_id.startswith("upload-"):
        path = UPLOAD_DIR / f"{dataset_id[len('upload-'):]}.csv"
        # the id is built by _slug(), so it can only name a file directly inside the uploads folder
        if re.fullmatch(r"upload-[a-z0-9-]+", dataset_id) and path.is_file():
            name = path.stem.replace("-", " ")
            return path, dict(id=dataset_id, source="upload", name=name, unit="rows",
                              tagline=None, positive_is="neutral", miss_note=None,
                              description=f"Uploaded file {path.name}.")
    raise DataError("That dataset is no longer available. Pick another one.")


def _parse(source) -> pd.DataFrame:
    """Read a CSV file (path or bytes). The separator is detected; '?' and blanks are missing values."""
    last_error = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            handle = io.BytesIO(source) if isinstance(source, bytes) else source
            frame = pd.read_csv(handle, sep=None, engine="python", encoding=encoding,
                                na_values=["?", " "], skipinitialspace=True)
            break
        except UnicodeDecodeError as error:
            last_error = error
        except Exception as error:
            raise DataError(f"This file could not be read as a table ({error}).")
    else:
        raise DataError(f"This file could not be read as text ({last_error}).")

    frame.columns = [str(name).strip() for name in frame.columns]
    if len(set(frame.columns)) != len(frame.columns):
        raise DataError("Two columns have the same name. Rename one of them and upload again.")
    frame = frame.dropna(how="all").reset_index(drop=True)
    return frame


def _version(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}-{stat.st_size}"


def load_frame(dataset_id: str):
    """The dataset as a DataFrame, read once per file version."""
    path, info = _locate(dataset_id)
    version = _version(path)
    with _lock:
        cached = _frames.get(dataset_id)
        if cached and cached[0] == version:
            return cached[1], info, version
    frame = _parse(path)
    with _lock:
        _frames[dataset_id] = (version, frame)
    return frame, info, version


def list_datasets():
    items = []
    for dataset_id in SAMPLES:
        if SAMPLES[dataset_id]["file"].is_file():
            items.append(dataset_id)
    if UPLOAD_DIR.is_dir():
        items += sorted(f"upload-{path.stem}" for path in UPLOAD_DIR.glob("*.csv"))
    out = []
    for dataset_id in items:
        try:
            frame, info, _ = load_frame(dataset_id)
        except DataError:
            continue
        out.append(dict(id=dataset_id, name=info["name"], source=info["source"],
                        description=info["description"], rows=len(frame), columns=frame.shape[1]))
    return out


def save_upload(filename: str, content: bytes) -> str:
    """Check an uploaded CSV, store it, and return its dataset id."""
    if not content:
        raise DataError("The file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise DataError(f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    frame = _parse(content)
    if frame.shape[1] < 2:
        raise DataError("The file needs at least two columns: one to predict and one to predict it from.")
    if frame.shape[1] > MAX_COLUMNS:
        raise DataError(f"The file has more than {MAX_COLUMNS} columns.")
    if len(frame) < MIN_ROWS:
        raise DataError(f"The file has only {len(frame)} rows. At least {MIN_ROWS} are needed.")
    if len(frame) > MAX_ROWS:
        raise DataError(f"The file has more than {MAX_ROWS:,} rows.")
    if not _target_candidates(frame):
        raise DataError(f"No column has between 2 and {MAX_CLASSES} distinct values, so there is nothing to classify.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slug(filename)
    (UPLOAD_DIR / f"{slug}.csv").write_bytes(content)
    return f"upload-{slug}"


def delete_upload(dataset_id: str):
    path, info = _locate(dataset_id)
    if info["source"] != "upload":
        raise DataError("The sample datasets cannot be removed.")
    path.unlink()
    with _lock:
        _frames.pop(dataset_id, None)
        for key in [key for key, problem in _problems.items() if problem.dataset["id"] == dataset_id]:
            del _problems[key]


# ── understanding the columns ────────────────────────────────────────────────

def as_text(series: pd.Series) -> pd.Series:
    """Values as category text: whole numbers lose their '.0', missing stays missing."""
    def convert(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        return str(value).strip()
    return series.map(convert).astype(object)


def _whole_numbers(values: pd.Series) -> bool:
    numbers = values.to_numpy(dtype=float)
    return bool(np.all(np.mod(numbers, 1) == 0))


def _looks_like_id(name: str, values: pd.Series) -> bool:
    named_id = re.search(r"(^|[^a-z])(id|index|no|number)($|[^a-z])|^unnamed|id$", name.lower()) is not None
    return named_id or values.is_monotonic_increasing or values.is_monotonic_decreasing


def profile_column(name: str, series: pd.Series, n_rows: int) -> dict:
    """Decide how a column is used: as a number, as a category, or not at all."""
    present = series.dropna()
    distinct = int(present.nunique())
    column = dict(name=name, missing=int(n_rows - len(present)), distinct=distinct, kind="category", reason=None)

    if len(present) == 0:
        column.update(kind="ignored", reason="empty")
    elif distinct == 1:
        column.update(kind="ignored", reason="same value in every row")
    elif pd.api.types.is_bool_dtype(series):
        pass
    elif pd.api.types.is_numeric_dtype(series):
        whole = _whole_numbers(present)
        if whole and distinct == len(present) == n_rows and _looks_like_id(name, present):
            column.update(kind="ignored", reason="looks like a row number or ID")
        elif whole and distinct <= CODE_VALUES:
            pass  # a few whole numbers are codes, not amounts
        else:
            column["kind"] = "number"
    elif distinct > MAX_CATEGORIES or (distinct > 20 and distinct / len(present) > 0.5):
        column.update(kind="ignored", reason="too many different values (names, IDs or free text)")
    return column


def _target_candidates(frame: pd.DataFrame):
    """Columns that could be the thing to predict: between 2 and MAX_CLASSES distinct values."""
    candidates = []
    for name in frame.columns:
        distinct = frame[name].nunique(dropna=True)
        if 2 <= distinct <= MAX_CLASSES:
            candidates.append(name)
    return candidates


_TARGET_NAMES = ["target", "class", "label", "outcome", "y", "diagnosis", "survived", "churn",
                 "result", "status", "species", "type", "default"]
_POSITIVE_WORDS = {"1", "yes", "true", "positive", "y", "t"}


def _guess_target(frame: pd.DataFrame, candidates):
    by_name = {name.lower(): name for name in candidates}
    for wanted in _TARGET_NAMES:
        if wanted in by_name:
            return by_name[wanted]
    if frame.columns[-1] in candidates:
        return frame.columns[-1]
    binary = [name for name in candidates if frame[name].nunique(dropna=True) == 2]
    return (binary or candidates)[0]


def _guess_positive(counts: pd.Series) -> str:
    """Which outcome to detect when the user has not said: "yes"/"1"/"true", else the rarer one."""
    named = [value for value in counts.index if value.lower() in _POSITIVE_WORDS]
    if len(named) == 1:
        return named[0]
    return counts.sort_values(kind="stable").index[0]


# ── a prepared problem ───────────────────────────────────────────────────────

@dataclass
class Problem:
    key: str                  # identifies dataset version + target + outcome (used for caching)
    dataset: dict             # id, name, source, description, unit, tagline, positive_is, miss_note
    target: str
    positive: str             # the raw class value the app tries to detect
    class_names: list         # readable names: [everything else, the outcome to detect]
    frame: pd.DataFrame       # the rows that have a target value, with all original columns
    n_rows: int               # rows in the file, including those without a target value
    columns: list             # profile of every column
    targets: list             # columns that could be predicted, each with its classes
    numeric: list
    categorical: list
    X: pd.DataFrame
    y: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def features(self):
        return self.numeric + self.categorical


def prepare(dataset_id: str, target: str | None = None, positive: str | None = None) -> Problem:
    """Build (or fetch from the cache) the problem for a dataset, target column and outcome."""
    frame, info, version = load_frame(dataset_id)
    sample = SAMPLES.get(dataset_id, {})

    candidates = _target_candidates(frame)
    if not candidates:
        raise DataError(f"No column has between 2 and {MAX_CLASSES} distinct values, so there is nothing to classify.")
    if target is None:
        target = sample.get("target") if sample.get("target") in candidates else _guess_target(frame, candidates)
    if target not in frame.columns:
        raise DataError(f"There is no column called '{target}'.")
    if target not in candidates:
        raise DataError(f"'{target}' cannot be predicted here: it needs between 2 and {MAX_CLASSES} distinct values.")

    labels_of = sample.get("labels", {}) if target == sample.get("target") else {}
    rows = frame[frame[target].notna()].reset_index(drop=True)
    classes = as_text(rows[target])
    counts = classes.value_counts()

    if positive is None:
        positive = sample["positive"] if target == sample.get("target") else _guess_positive(counts)
    if positive not in counts.index:
        raise DataError(f"'{positive}' is not a value of the column '{target}'.")

    key = f"{dataset_id}|{version}|{target}|{positive}"
    with _lock:
        if key in _problems:
            _problems.move_to_end(key)
            return _problems[key]

    y = (classes == positive).astype(int)
    positive_name = labels_of.get(positive, positive)
    others = [value for value in counts.index if value != positive]
    negative_name = labels_of.get(others[0], others[0]) if len(others) == 1 else f"not {positive_name}"

    if len(rows) < MIN_ROWS:
        raise DataError(f"Only {len(rows)} rows have a value in '{target}'. At least {MIN_ROWS} are needed.")
    for name, count in ((positive_name, int(y.sum())), (negative_name, int((1 - y).sum()))):
        if count < MIN_PER_CLASS:
            raise DataError(f"Only {count} rows are '{name}'. At least {MIN_PER_CLASS} rows of each outcome are needed.")

    columns = [profile_column(name, frame[name], len(frame)) for name in frame.columns]
    for column in columns:
        if column["name"] == target:
            column.update(kind="target", reason=None)
    numeric = [c["name"] for c in columns if c["kind"] == "number"]
    categorical = [c["name"] for c in columns if c["kind"] == "category"]
    if not numeric and not categorical:
        raise DataError("None of the other columns can be used to predict with.")

    X = pd.DataFrame(index=rows.index)
    for name in numeric:
        X[name] = pd.to_numeric(rows[name], errors="coerce").astype(float)
    for name in categorical:
        X[name] = as_text(rows[name])
    X = X[numeric + categorical]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    targets = []
    for name in candidates:
        values = as_text(frame[name].dropna()).value_counts()
        names = sample.get("labels", {}) if name == sample.get("target") else {}
        targets.append(dict(name=name, classes=[
            dict(value=value, label=names.get(value, value), count=int(count))
            for value, count in values.items()]))

    problem = Problem(
        key=key, dataset=info, target=target, positive=positive,
        class_names=[negative_name, positive_name], frame=rows, n_rows=len(frame),
        columns=columns, targets=targets,
        numeric=numeric, categorical=categorical, X=X, y=y,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    )
    with _lock:
        _problems[key] = problem
        while len(_problems) > 6:
            _problems.popitem(last=False)
    return problem
