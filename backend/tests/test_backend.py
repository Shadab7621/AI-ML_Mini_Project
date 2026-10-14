"""Tests for the data handling, the models and the HTTP API.

Run from the project root:   python -m pytest backend/tests
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import datasets, explore, main, ml
from backend.datasets import DataError
from backend.tests.conftest import csv_bytes

SAMPLE_IDS = list(datasets.SAMPLES)
# the models are quick on every dataset except these combinations, which are skipped
BIG = {"telco", "mushrooms"}


# ── the sample datasets ──────────────────────────────────────────────────────

def test_every_sample_file_exists():
    for sample_id, sample in datasets.SAMPLES.items():
        assert sample["file"].is_file(), f"{sample_id}: {sample['file']} is missing"


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_sample_is_understood(sample_id):
    problem = datasets.prepare(sample_id)
    sample = datasets.SAMPLES[sample_id]

    assert problem.target == sample["target"]
    assert problem.positive == sample["positive"]
    assert len(problem.class_names) == 2
    assert problem.X.shape[0] == len(problem.y) == len(problem.frame)
    assert problem.features, "no usable feature columns"
    assert set(problem.y.unique()) == {0, 1}
    assert len(problem.X_train) + len(problem.X_test) == len(problem.X)
    assert abs(len(problem.X_test) / len(problem.X) - datasets.TEST_SIZE) < 0.02
    # both outcomes in both parts, so every score is defined
    assert problem.y_train.nunique() == 2 and problem.y_test.nunique() == 2
    # the column to predict never leaks into the features
    assert problem.target not in problem.features
    # numeric features really are numbers
    assert all(pd.api.types.is_float_dtype(problem.X[name]) for name in problem.numeric)


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_sample_summary(sample_id):
    problem = datasets.prepare(sample_id)
    summary = main.clean(explore.summarise(problem))

    assert summary["n_samples"] == len(problem.X)
    assert sum(item["count"] for item in summary["class_counts"]) == len(problem.X)
    labels = summary["association"]["labels"]
    matrix = summary["association"]["matrix"]
    assert labels[0] == problem.target
    assert len(matrix) == len(labels) and all(len(row) == len(labels) for row in matrix)
    values = [v for row in matrix for v in row if v is not None]
    assert all(0 <= v <= 1.0001 for v in values), "association strengths must be between 0 and 1"
    assert 1 <= len(summary["top_features"]) <= explore.TOP_FEATURES
    for item in summary["top_features"]:
        assert 0 <= item["strength"] <= 1.0001
        assert sum(v["negative"] + v["positive"] for v in item["values"]) <= len(problem.X)
    assert len(summary["fields"]) == len(problem.features)
    assert len(summary["preview"]["rows"]) == min(explore.PREVIEW_ROWS, len(problem.frame))


@pytest.mark.parametrize("sample_id", [s for s in SAMPLE_IDS if s not in BIG])
def test_every_model_trains_on_small_samples(sample_id):
    problem = datasets.prepare(sample_id)
    for spec in ml.CLASSIFIERS:
        params = ml.resolve_params(spec["name"], {})
        pipe = ml.fit_pipeline(problem, spec["name"], params)
        result = main.clean(ml.evaluate(problem, pipe))

        assert 0 <= result["metrics"]["Accuracy"] <= 1
        matrix = np.array(result["confusion_matrix"])
        assert matrix.shape == (2, 2) and matrix.sum() == len(problem.y_test)
        assert len(result["y_pred"]) == len(problem.y_test)
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(problem.X_test)
            assert np.allclose(proba.sum(axis=1), 1)
        # a model that learns something beats guessing the commonest class
        majority = max(problem.y_test.mean(), 1 - problem.y_test.mean())
        assert result["metrics"]["Accuracy"] >= majority - 0.15, f"{spec['name']} on {sample_id}"


@pytest.mark.parametrize("sample_id", sorted(BIG))
def test_big_samples_train_with_quick_models(sample_id):
    problem = datasets.prepare(sample_id)
    for name in ("Logistic Regression", "Decision Tree", "Naive Bayes"):
        pipe = ml.fit_pipeline(problem, name, ml.resolve_params(name, {}))
        assert ml.evaluate(problem, pipe)["metrics"]["Accuracy"] > 0.6


@pytest.mark.parametrize("sample_id", ["iris", "penguins", "diabetes", "tictactoe"])
def test_analyses_work(sample_id):
    problem = datasets.prepare(sample_id)
    for name in ("Logistic Regression", "Random Forest", "Naive Bayes"):
        pipe = ml.fit_pipeline(problem, name, ml.resolve_params(name, {}))
        importance = ml.feature_importance(problem, pipe, name)
        assert 1 <= len(importance["items"]) <= 15
        assert {item["feature"] for item in importance["items"]} <= set(problem.features)
        cv = ml.cross_validation(problem, pipe)
        assert len(cv["scores"]) == 5 and 0 <= cv["mean"] <= 1
        curve = ml.learning_curve_data(problem, pipe)
        assert len(curve["sizes"]) == 9


def test_prediction_of_one_example():
    problem = datasets.prepare("titanic")
    pipe = ml.fit_pipeline(problem, "Logistic Regression", ml.resolve_params("Logistic Regression", {}))
    # a first-class woman against a third-class man
    woman = ml.predict_one(problem, pipe, {"Pclass": "1", "Sex": "female", "Age": 30, "Fare": 80})
    man = ml.predict_one(problem, pipe, {"Pclass": "3", "Sex": "male", "Age": 30, "Fare": 8})
    assert woman["probability"] > man["probability"]
    # nothing filled in at all is allowed
    assert ml.predict_one(problem, pipe, {})["label"] in problem.class_names
    # a category the model has never seen is allowed too
    assert ml.predict_one(problem, pipe, {"Embarked": "Z"})["label"] in problem.class_names
    with pytest.raises(DataError):
        ml.predict_one(problem, pipe, {"Age": "old"})


def test_settings_are_checked():
    assert ml.resolve_params("Decision Tree", {})["max_depth"] == 10
    for bad in ({"max_depth": 0}, {"max_depth": 99}, {"max_depth": "deep"}, {"criterion": "nope"}):
        with pytest.raises(ValueError):
            ml.resolve_params("Decision Tree", bad)
    with pytest.raises(ValueError):
        ml.resolve_params("No such model", {})


def test_results_do_not_depend_on_row_order_of_the_file(uploads):
    """The same data in a different order must give the same split sizes and class balance."""
    source = pd.read_csv(datasets.SAMPLES["heart"]["file"])
    shuffled = source.sample(frac=1, random_state=1)
    first = datasets.save_upload("a.csv", source.to_csv(index=False).encode())
    second = datasets.save_upload("b.csv", shuffled.to_csv(index=False).encode())
    a = datasets.prepare(first, "heart_disease", "yes")
    b = datasets.prepare(second, "heart_disease", "yes")
    assert (len(a.X_train), len(a.X_test)) == (len(b.X_train), len(b.X_test))
    assert a.y.sum() == b.y.sum()


# ── uploaded files ───────────────────────────────────────────────────────────

def rows(n=120, **extra):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "income": rng.normal(50000, 12000, n).round(),
        "city": rng.choice(["north", "south", "east"], n),
    })
    frame["label"] = np.where(frame["income"] + rng.normal(0, 8000, n) > 50000, "yes", "no")
    for key, value in extra.items():
        frame[key] = value
    return frame


def test_upload_roundtrip(uploads):
    dataset_id = datasets.save_upload("My Data (v2).csv", rows().to_csv(index=False).encode())
    assert dataset_id == "upload-my-data-v2"
    problem = datasets.prepare(dataset_id)
    assert problem.target == "label"
    assert set(problem.numeric) == {"age", "income"} and problem.categorical == ["city"]
    assert any(item["id"] == dataset_id for item in datasets.list_datasets())
    datasets.delete_upload(dataset_id)
    assert not any(item["id"] == dataset_id for item in datasets.list_datasets())
    with pytest.raises(DataError):
        datasets.prepare(dataset_id)


@pytest.mark.parametrize("separator", [",", ";", "\t", "|"])
def test_separators_are_detected(uploads, separator):
    content = rows().to_csv(index=False, sep=separator).encode()
    problem = datasets.prepare(datasets.save_upload("x.csv", content))
    assert problem.target == "label" and len(problem.X) == 120


def test_byte_order_mark_and_latin1(uploads):
    frame = rows()
    frame["city"] = frame["city"].replace({"north": "nörd"})
    bom = b"\xef\xbb\xbf" + frame.to_csv(index=False).encode("utf-8")
    assert datasets.prepare(datasets.save_upload("bom.csv", bom)).target == "label"
    latin = frame.to_csv(index=False).encode("latin-1")
    assert datasets.prepare(datasets.save_upload("latin.csv", latin)).target == "label"


def test_messy_values_are_tolerated(uploads):
    frame = rows()
    frame["age"] = frame["age"].astype(float)
    frame.loc[3, "income"] = np.nan
    frame.loc[5, "age"] = np.inf
    frame.loc[7, "city"] = None
    frame.loc[9, "label"] = None          # a row without an answer is dropped
    frame["notes"] = "same"               # constant column
    frame["row_id"] = np.arange(len(frame))
    frame["name"] = [f"person {i}" for i in range(len(frame))]
    problem = datasets.prepare(datasets.save_upload("messy.csv", frame.to_csv(index=False).encode()))
    assert len(problem.X) == len(frame) - 1
    assert np.isfinite(problem.X[problem.numeric].fillna(0).to_numpy()).all()
    ignored = {c["name"] for c in problem.columns if c["kind"] == "ignored"}
    assert {"notes", "row_id", "name"} <= ignored
    # and a model can still be trained on it
    pipe = ml.fit_pipeline(problem, "Random Forest", ml.resolve_params("Random Forest", {}))
    assert ml.evaluate(problem, pipe)["metrics"]["Accuracy"] > 0.5


def test_numeric_class_column(uploads):
    frame = rows()
    frame["label"] = (frame["label"] == "yes").astype(int)
    problem = datasets.prepare(datasets.save_upload("numeric.csv", frame.to_csv(index=False).encode()))
    assert problem.target == "label" and problem.positive == "1"


def test_more_than_two_classes(uploads):
    frame = rows()
    frame["grade"] = pd.cut(frame["income"], 3, labels=["low", "mid", "high"]).astype(str)
    frame = frame.drop(columns=["label"])
    dataset_id = datasets.save_upload("grades.csv", frame.to_csv(index=False).encode())
    problem = datasets.prepare(dataset_id, "grade", "high")
    assert problem.class_names[1] == "high" and problem.class_names[0].startswith("not")
    assert problem.y.sum() == (frame["grade"] == "high").sum()


@pytest.mark.parametrize("name, content, message", [
    ("empty.csv", b"", "empty"),
    ("tiny.csv", csv_bytes("a,b\n1,x\n2,y\n3,x"), "only 3 rows"),
    ("onecol.csv", csv_bytes("a\n" + "\n".join(str(i) for i in range(80))), "at least two columns"),
    ("numbers.csv", csv_bytes("a,b\n" + "\n".join(f"{i * 0.37},{i * 1.9}" for i in range(80))), "nothing to classify"),
])
def test_unusable_files_are_refused_with_a_reason(uploads, name, content, message):
    with pytest.raises(DataError, match=message):
        datasets.save_upload(name, content)


def test_too_few_of_one_outcome(uploads):
    frame = rows(100)
    frame["label"] = ["rare"] * 4 + ["common"] * 96
    dataset_id = datasets.save_upload("rare.csv", frame.to_csv(index=False).encode())
    with pytest.raises(DataError, match="At least 10 rows of each outcome"):
        datasets.prepare(dataset_id)


def test_duplicate_column_names_are_made_unique(uploads):
    lines = ["a,a,label"] + [f"{(i * 7) % 80},{i * 3 % 7},{'yes' if i % 2 else 'no'}" for i in range(80)]
    content = "\n".join(lines).encode()
    problem = datasets.prepare(datasets.save_upload("dup.csv", content))
    assert sorted(problem.features) == ["a", "a.1"]


def test_size_limit(uploads):
    big = b"a,label\n" + b"1,yes\n" * (datasets.MAX_UPLOAD_BYTES // 6 + 10)
    with pytest.raises(DataError, match="larger than"):
        datasets.save_upload("big.csv", big)


# ── the HTTP API ─────────────────────────────────────────────────────────────

@pytest.fixture()
def client(uploads):
    return TestClient(main.app, raise_server_exceptions=False)


def test_api_lists_models_and_datasets(client):
    meta = client.get("/api/meta").json()
    assert [c["short"] for c in meta["classifiers"]] == [
        "SVM", "Logistic Reg.", "Random Forest", "KNN", "Decision Tree", "Gradient Boosting", "Naive Bayes"]
    listed = client.get("/api/datasets").json()
    ids = {item["id"] for item in listed["datasets"]}
    assert set(SAMPLE_IDS) <= ids
    assert all(item["rows"] > 0 for item in listed["datasets"])


def test_api_train_predict_and_analyses(client):
    body = {"dataset": "heart", "classifier": "Logistic Regression", "params": {}}
    problem = client.post("/api/problem", json={"dataset": "heart"}).json()
    assert problem["target"] == "heart_disease"
    body.update(target=problem["target"], positive=problem["positive"])

    trained = client.post("/api/train", json=body)
    assert trained.status_code == 200
    assert trained.json()["metrics"]["Accuracy"] > 0.7
    assert client.post("/api/train", json=body).json() == trained.json()      # cached, identical

    example = client.post("/api/example", json={**body, "index": 0}).json()
    answer = client.post("/api/predict", json={**body, "values": example["values"]}).json()
    assert answer["label"] in problem["class_names"] and 0 <= answer["probability"] <= 1

    for path in ("/api/cross-validation", "/api/feature-importance", "/api/learning-curve"):
        assert client.post(path, json=body).status_code == 200, path
    compared = client.post("/api/compare", json={"dataset": "heart"}).json()
    assert len(compared["rows"]) == 7 and compared["best"] in {r["model"] for r in compared["rows"]}


def test_api_upload_flow(client):
    content = rows().to_csv(index=False).encode()
    uploaded = client.post("/api/datasets", content=content, headers={"x-filename": "people.csv"})
    assert uploaded.status_code == 200
    dataset_id = uploaded.json()["id"]
    assert client.post("/api/problem", json={"dataset": dataset_id}).status_code == 200
    assert client.delete(f"/api/datasets/{dataset_id}").json() == {"deleted": dataset_id}
    assert client.post("/api/problem", json={"dataset": dataset_id}).status_code == 422


def test_api_refusals_are_readable(client):
    def detail(response):
        assert response.status_code == 422, response.text
        return response.json()["detail"]

    assert "no longer available" in detail(client.post("/api/problem", json={"dataset": "nope"}))
    assert "no column called" in detail(client.post("/api/problem", json={"dataset": "titanic", "target": "X"}))
    assert "cannot be predicted" in detail(client.post("/api/problem", json={"dataset": "titanic", "target": "Name"}))
    assert "not a value" in detail(client.post("/api/problem", json={"dataset": "titanic", "positive": "7"}))
    assert "between 1 and 30" in detail(client.post(
        "/api/train", json={"dataset": "titanic", "classifier": "Decision Tree", "params": {"max_depth": 99}}))
    assert "must be a number" in detail(client.post(
        "/api/predict", json={"dataset": "titanic", "classifier": "Decision Tree", "params": {}, "values": {"Age": "old"}}))
    assert "CSV file" in detail(client.post("/api/datasets", content=b"x", headers={"x-filename": "a.png"}))
    assert "cannot be removed" in detail(client.delete("/api/datasets/mushrooms"))


def test_api_path_tricks_do_not_escape(client):
    assert client.delete("/api/datasets/upload-..%2F..%2FMushrooms").status_code in (404, 422)
    assert datasets.SAMPLES["mushrooms"]["file"].is_file()
    assert "no longer available" in client.post("/api/problem", json={"dataset": "upload-../../Mushrooms"}).json()["detail"]


def test_api_unexpected_errors_still_return_a_message(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("something odd")
    monkeypatch.setattr(ml, "evaluate", boom)
    response = client.post("/api/train", json={"dataset": "iris", "classifier": "Naive Bayes", "params": {}})
    assert response.status_code == 500
    assert "RuntimeError" in response.json()["detail"]


# ── things that used to go wrong ─────────────────────────────────────────────

def test_learning_curve_survives_slices_with_one_outcome_only():
    """Small slices of the wine data can hold a single outcome; the curve must still be drawn."""
    problem = datasets.prepare("wine")
    for name in ("Logistic Regression", "Support Vector Machine (SVM)"):
        pipe = ml.fit_pipeline(problem, name, ml.resolve_params(name, {}))
        curve = ml.learning_curve_data(problem, pipe)
        assert len(curve["sizes"]) >= 2
        assert all(0 <= value <= 1 for value in curve["cv_mean"])


def test_analyses_asked_for_together_do_not_break_each_other():
    """The Charts page asks for importance and the learning curve at the same moment."""
    from concurrent.futures import ThreadPoolExecutor

    def request(**extra):
        return main.ModelRequest(dataset="breast_cancer", classifier="Logistic Regression", params={}, **extra)

    calls = [main.feature_importance, main.learning_curve, main.cross_validation] * 2
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = [pool.submit(call, request()) for call in calls]
        results = [future.result(timeout=300) for future in futures]
    assert len(results) == 6 and all(results)
