import type { MetricKey } from "@/lib/api"
import { downloadCsv } from "@/lib/download"
import { useApp } from "@/state/app-state"

/** Column order of the comparison table and its CSV. */
export const METRIC_ORDER: MetricKey[] = [
  "Accuracy",
  "Precision",
  "Recall",
  "F1 Score",
  "MCC",
  "Cohen κ",
  "ROC-AUC",
  "Avg Precision",
]

/** The things buttons do, shared by the sidebar and the pages. */
export function useActions() {
  const { train, compare, getTab, setTab, result, comparison } = useApp()

  const trainAndShow = () => {
    train()
    const tab = getTab()
    if (tab === "dataset" || tab === "compare") setTab("results")
  }

  const compareAndShow = () => {
    compare()
    setTab("compare")
  }

  const trained = result.data
  const exportPredictions = trained
    ? () =>
        downloadCsv("predictions.csv", [
          ["True Label", "Predicted", "Correct"],
          ...trained.y_true.map((truth, i) => [
            trained.class_names[truth],
            trained.class_names[trained.y_pred[i]],
            truth === trained.y_pred[i] ? "yes" : "no",
          ]),
        ])
    : undefined

  const compared = comparison.data
  const exportComparison = compared
    ? () =>
        downloadCsv("model_comparison.csv", [
          ["Model", ...METRIC_ORDER],
          ...compared.rows.map((row) => [
            row.model,
            ...METRIC_ORDER.map((key) => row.metrics[key] ?? ""),
          ]),
        ])
    : undefined

  return { trainAndShow, compareAndShow, exportPredictions, exportComparison }
}
