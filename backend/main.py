"""HTTP API for the React dashboard.

Run from the project root:
    python -m uvicorn backend.main:app --port 8000
"""
import json
import math
import random
import threading
import time
from collections import OrderedDict
from urllib.parse import unquote

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from . import datasets, explore, ml
from .datasets import DataError

app = FastAPI(title="Classification Visualiser API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(DataError)
def data_error(_: Request, error: DataError):
    """Problems the user can fix come back as a readable message, not a crash."""
    return JSONResponse(status_code=422, content={"detail": str(error)})


# ── helpers ──────────────────────────────────────────────────────────────────

def clean(obj):
    """Turn numpy values into plain JSON types; NaN and infinity become null."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# Results are remembered per (problem, classifier, settings), so asking twice for
# the same thing never trains twice. Only the most recent entries are kept.
_CACHE_SIZE = 64
_cache: "OrderedDict[tuple, object]" = OrderedDict()
_locks: dict = {}
_guard = threading.Lock()


def cached(kind, key, compute):
    k = (kind, key)
    with _guard:
        lock = _locks.setdefault(k, threading.Lock())
    with lock:
        with _guard:
            if k in _cache:
                _cache.move_to_end(k)
                return _cache[k]
        value = compute()
        with _guard:
            _cache[k] = value
            while len(_cache) > _CACHE_SIZE:
                old, _ = _cache.popitem(last=False)
                _locks.pop(old, None)
        return value


class ProblemRequest(BaseModel):
    """Which data, which column to predict, and which outcome to detect.
    Leave target and positive empty to let the server choose sensible defaults."""
    dataset: str
    target: str | None = None
    positive: str | None = None


class ModelRequest(ProblemRequest):
    classifier: str
    params: dict = {}


class PredictRequest(ModelRequest):
    values: dict = {}


class ExampleRequest(ProblemRequest):
    index: int | None = None


def problem_of(req: ProblemRequest):
    return datasets.prepare(req.dataset, req.target, req.positive)


def resolve(req: ModelRequest):
    problem = problem_of(req)
    try:
        params = ml.resolve_params(req.classifier, req.params)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
    key = json.dumps([problem.key, req.classifier, params], sort_keys=True)
    return problem, req.classifier, params, key


def fitted(problem, name, params, key):
    def train():
        start = time.perf_counter()
        pipe = ml.fit_pipeline(problem, name, params)
        return pipe, time.perf_counter() - start
    return cached("fit", key, train)


# ── datasets ─────────────────────────────────────────────────────────────────

@app.get("/api/meta")
def meta():
    return {"classifiers": [{**c, "scaled": c["name"] in ml.NEEDS_SCALING} for c in ml.CLASSIFIERS]}


@app.get("/api/datasets")
def list_datasets():
    return {"datasets": datasets.list_datasets(),
            "max_upload_mb": datasets.MAX_UPLOAD_BYTES // (1024 * 1024)}


@app.post("/api/datasets")
async def upload_dataset(request: Request):
    """Upload a CSV file. The file is the request body; its name goes in the X-Filename header."""
    filename = unquote(request.headers.get("x-filename", "data.csv"))
    if not filename.lower().endswith((".csv", ".txt", ".data", ".tsv")):
        raise DataError("Please upload a CSV file (a table saved as text).")
    declared = int(request.headers.get("content-length") or 0)
    if declared > datasets.MAX_UPLOAD_BYTES:
        raise DataError(f"The file is larger than {datasets.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    content = await request.body()
    dataset_id = await run_in_threadpool(datasets.save_upload, filename, content)
    return {"id": dataset_id}


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    datasets.delete_upload(dataset_id)
    return {"deleted": dataset_id}


@app.post("/api/problem")
def describe_problem(req: ProblemRequest):
    """Everything the Dataset page shows, for the chosen data, target column and outcome."""
    prepared = problem_of(req)
    return cached("summary", prepared.key, lambda: clean(explore.summarise(prepared)))


@app.post("/api/example")
def example(req: ExampleRequest):
    """A row of the test set (a random one unless an index is given), to try the model on."""
    prepared = problem_of(req)
    index = req.index if req.index is not None else random.randrange(len(prepared.X_test))
    return clean(explore.example_row(prepared, index))


# ── models ───────────────────────────────────────────────────────────────────

@app.post("/api/train")
def train(req: ModelRequest):
    problem, name, params, key = resolve(req)

    def compute():
        pipe, seconds = fitted(problem, name, params, key)
        out = ml.evaluate(problem, pipe)
        out.update(classifier=name, params=params, train_seconds=seconds)
        return clean(out)
    return cached("train", key, compute)


@app.post("/api/feature-importance")
def feature_importance(req: ModelRequest):
    problem, name, params, key = resolve(req)
    return cached("importance", key, lambda: clean(
        ml.feature_importance(problem, fitted(problem, name, params, key)[0], name)))


@app.post("/api/learning-curve")
def learning_curve(req: ModelRequest):
    problem, name, params, key = resolve(req)
    return cached("learning", key, lambda: clean(
        ml.learning_curve_data(problem, fitted(problem, name, params, key)[0])))


@app.post("/api/cross-validation")
def cross_validation(req: ModelRequest):
    problem, name, params, key = resolve(req)
    return cached("cv", key, lambda: clean(
        ml.cross_validation(problem, fitted(problem, name, params, key)[0])))


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Classify one example with the trained model (trained on demand if it is not in memory)."""
    problem, name, params, key = resolve(req)
    return clean(ml.predict_one(problem, fitted(problem, name, params, key)[0], req.values))


@app.post("/api/compare")
def compare(req: ProblemRequest):
    prepared = problem_of(req)

    def compute():
        df = ml.compare_all_models(prepared)
        best = df["F1 Score"].idxmax()
        return clean({
            "rows": [{"model": model, "metrics": row.to_dict()} for model, row in df.iterrows()],
            "best": best,
        })
    return cached("compare", prepared.key, compute)
