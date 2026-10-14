// Typed client for the Python API (see backend/main.py).

export type ParamSpec =
  | {
      key: string
      label: string
      type: "int" | "float"
      min: number
      max: number
      step: number
      default: number
      help?: string
      /** Set when the setting only matters for one kind of column. */
      applies_to?: "number" | "category"
    }
  | {
      key: string
      label: string
      type: "choice"
      options: string[]
      default: string
      help?: string
    }

export interface ClassifierSpec {
  name: string
  short: string
  scaled: boolean
  params: ParamSpec[]
}

export interface Meta {
  classifiers: ClassifierSpec[]
}

export type ParamValues = Record<string, number | string>

export interface DatasetInfo {
  id: string
  name: string
  source: "sample" | "upload"
  description: string
  rows: number
  columns: number
}

export interface DatasetList {
  datasets: DatasetInfo[]
  max_upload_mb: number
}

/** Which data, which column to predict, which outcome to detect. `null` lets the server choose. */
export interface ProblemSpec {
  dataset: string
  target: string | null
  positive: string | null
}

export interface ModelRequest extends ProblemSpec {
  classifier: string
  params: ParamValues
}

export interface ColumnProfile {
  name: string
  kind: "number" | "category" | "ignored" | "target"
  reason: string | null
  missing: number
  distinct: number
}

/** What the "try one example" form needs to know about a feature. */
export type Field =
  | { name: string; kind: "number"; min: number; max: number; default: number; whole: boolean }
  | { name: string; kind: "category"; options: string[]; default: string }

/** Everything the Dataset page shows for the chosen data, target and outcome. */
export interface Problem {
  dataset: {
    id: string
    name: string
    source: "sample" | "upload"
    description: string
    /** What one row is, in the plural: "mushrooms", "patients", "rows". */
    unit: string
    tagline: string | null
    /** Whether the outcome being detected is the bad one, the good one, or neither. */
    positive_is: "bad" | "good" | "neutral"
    miss_note: string | null
  }
  target: string
  positive: string
  /** Readable names: [everything else, the outcome being detected]. */
  class_names: [string, string]
  targets: { name: string; classes: { value: string; label: string; count: number }[] }[]
  /** Rows in the file. */
  n_rows: number
  /** Rows that have a value in the target column: the ones the models can use. */
  n_samples: number
  n_features: number
  n_numeric: number
  n_categorical: number
  n_train: number
  n_test: number
  class_counts: { label: string; count: number }[]
  columns: ColumnProfile[]
  fields: Field[]
  /** How strongly each pair of columns is related, 0 to 1. The target comes first. */
  association: { labels: string[]; matrix: (number | null)[][]; hidden: number }
  top_features: {
    feature: string
    strength: number
    kind: "number" | "category"
    values: { value: string; negative: number; positive: number }[]
  }[]
  preview: { columns: string[]; rows: (string | number | boolean | null)[][] }
}

// Keys are the metric names the Python side uses.
export interface Metrics {
  Accuracy: number
  Precision: number
  Recall: number
  "F1 Score": number
  MCC: number
  "Cohen κ": number
  "ROC-AUC": number | null
  "Avg Precision": number | null
}
export type MetricKey = keyof Metrics

export interface ReportRow {
  precision: number
  recall: number
  "f1-score": number
  support: number
}

export interface TrainResult {
  classifier: string
  params: ParamValues
  train_seconds: number
  class_names: [string, string]
  metrics: Metrics
  /** Rows are the true class, columns the predicted class; index 1 is the outcome being detected. */
  confusion_matrix: [[number, number], [number, number]]
  /** Keys: "negative", "positive", "accuracy", "macro avg", "weighted avg". */
  report: Record<string, ReportRow | number>
  y_true: number[]
  y_pred: number[]
  roc: { fpr: number[]; tpr: number[] } | null
  pr: { recall: number[]; precision: number[] } | null
}

export interface FeatureImportance {
  method: "impurity" | "permutation"
  items: { feature: string; importance: number; std: number | null }[]
}

export interface LearningCurve {
  sizes: number[]
  train_mean: number[]
  train_std: number[]
  cv_mean: number[]
  cv_std: number[]
  gap: number
}

export interface CrossValidation {
  scores: number[]
  mean: number
  std: number
  min: number
  max: number
}

export interface Comparison {
  rows: { model: string; metrics: Metrics }[]
  best: string
}

export type ExampleValues = Record<string, number | string | null>

export interface Example {
  values: ExampleValues
  truth: string
}

export interface Prediction {
  label: string
  is_positive: boolean
  /** The model's estimate that this is the outcome being detected, 0 to 1. */
  probability: number | null
}

/** The server understood the request and said no (bad file, bad setting...). Retrying will not help. */
export class ApiError extends Error {}

const UNREACHABLE = "Can't reach the Python server. Start it with start.bat, then try again."
const BASE: string = import.meta.env.VITE_API_URL ?? ""

async function send<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(BASE + path, init)
  } catch {
    throw new Error(UNREACHABLE)
  }
  if (!response.ok) {
    let detail = ""
    try {
      const data = await response.json()
      detail = typeof data?.detail === "string" ? data.detail : ""
    } catch {
      // no JSON body — the Vite proxy answers with a bare 5xx when the server is down
    }
    if (!detail && response.status >= 500) throw new Error(UNREACHABLE)
    throw new ApiError(detail || `The server answered with an error (${response.status}).`)
  }
  return response.json() as Promise<T>
}

const post = <T>(path: string, body: unknown) =>
  send<T>(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  })

export const api = {
  meta: () => send<Meta>("/api/meta"),
  datasets: () => send<DatasetList>("/api/datasets"),
  upload: (file: File) =>
    send<{ id: string }>("/api/datasets", {
      method: "POST",
      headers: { "content-type": "text/csv", "x-filename": encodeURIComponent(file.name) },
      body: file,
    }),
  removeDataset: (id: string) =>
    send<{ deleted: string }>(`/api/datasets/${encodeURIComponent(id)}`, { method: "DELETE" }),
  problem: (spec: ProblemSpec) => post<Problem>("/api/problem", spec),
  example: (spec: ProblemSpec) => post<Example>("/api/example", spec),
  train: (req: ModelRequest) => post<TrainResult>("/api/train", req),
  featureImportance: (req: ModelRequest) => post<FeatureImportance>("/api/feature-importance", req),
  learningCurve: (req: ModelRequest) => post<LearningCurve>("/api/learning-curve", req),
  crossValidation: (req: ModelRequest) => post<CrossValidation>("/api/cross-validation", req),
  predict: (req: ModelRequest, values: ExampleValues) =>
    post<Prediction>("/api/predict", { ...req, values }),
  compare: (spec: ProblemSpec) => post<Comparison>("/api/compare", spec),
}
