
import { CurveChart, ImportanceChart, LearningCurveChart } from "@/components/charts/model-charts"
import { Panel, Pending, Reveal, Working } from "@/components/common"
import { ConfusionMatrix } from "@/components/confusion-matrix"
import { NeedsModel } from "@/components/needs-model"
import { ModelSummary, PageHeader } from "@/components/page-header"
import { Skeleton } from "@/components/ui/skeleton"
import { formatScore } from "@/lib/format"
import { WhenActive } from "@/state/active"
import { useApp } from "@/state/app-state"
import { cn } from "@/lib/utils"

const DESCRIPTION = "Where the trained model is right, where it is wrong, and what it relies on."

// One column on small widths, two on medium, three on wide.
const GRID = "grid grid-cols-1 gap-5 @3xl:grid-cols-2 @7xl:grid-cols-3"

function Unavailable({ what }: { what: string }) {
  return (
    <p className="grid min-h-[300px] place-items-center rounded-2xl bg-muted/40 px-6 text-center text-muted-foreground">
      This model does not give the scores needed for a {what}.
    </p>
  )
}

export function ChartsPage() {
  const { result, importance, learning, loadImportance, loadLearning } = useApp()


  const data = result.data
  if (!data) {
    return (
      <div className="flex flex-col gap-5">
        <PageHeader title="Charts" description={DESCRIPTION} />
        <NeedsModel
          what="its charts"
          skeleton={
            <div className={GRID}>
              {[0, 1, 2].map((i) => (
                <Skeleton key={i} className="h-[420px] rounded-[24px]" />
              ))}
            </div>
          }
        />
      </div>
    )
  }

  const { metrics } = data
  const auc = metrics["ROC-AUC"]
  const averagePrecision = metrics["Avg Precision"]
  // while a new model trains, the previous model's charts stay on screen, dimmed
  const panel = cn("h-full transition-opacity duration-300", result.status !== "ready" && "opacity-45")

  return (
    <div className="flex flex-col gap-5">
      <WhenActive load={loadImportance} />
      <WhenActive load={loadLearning} />
      <PageHeader title="Charts" description={DESCRIPTION} />
      <ModelSummary result={data} updating={result.status === "loading"} />

      <div className={GRID}>
        <Reveal index={0}>
          <Panel
            className={panel}
            title="Confusion matrix"
            subtitle="Test rows by what they really are and what the model said."
          >
            <ConfusionMatrix matrix={data.confusion_matrix} classNames={data.class_names} />
          </Panel>
        </Reveal>

        <Reveal index={1}>
          <Panel
            className={panel}
            title="ROC curve"
            subtitle={
              auc === null
                ? "How many real cases are caught against how many false alarms are raised."
                : `Area under the curve ${formatScore(auc)}. The dashed line is random guessing.`
            }
          >
            {data.roc ? (
              <CurveChart
                x={data.roc.fpr}
                y={data.roc.tpr}
                xLabel="False positive rate"
                yLabel="True positive rate"
                diagonal
                description="ROC curve: true positive rate against false positive rate."
              />
            ) : (
              <Unavailable what="ROC curve" />
            )}
          </Panel>
        </Reveal>

        <Reveal index={2}>
          <Panel
            className={panel}
            title="Precision-recall curve"
            subtitle={
              averagePrecision === null
                ? "Precision at every level of recall."
                : `Average precision ${formatScore(averagePrecision)}. Higher and flatter is better.`
            }
          >
            {data.pr ? (
              <CurveChart
                x={data.pr.recall}
                y={data.pr.precision}
                xLabel="Recall"
                yLabel="Precision"
                step
                description="Precision-recall curve: precision at every level of recall."
              />
            ) : (
              <Unavailable what="precision-recall curve" />
            )}
          </Panel>
        </Reveal>

        <Reveal index={3}>
          <Panel
            className="h-full"
            title="Learning curve"
            subtitle={
              learning.data
                ? `Accuracy as the model sees more data. Gap between the two lines at the end: ${formatScore(learning.data.gap, 3)}.`
                : "Accuracy as the model sees more training data."
            }
            actions={learning.data && learning.status === "loading" && <Working>Updating…</Working>}
          >
            <Pending
              resource={learning}
              height={356}
              waiting="Training the model on 9 growing slices of the data, 5 times each. Quick for most models; SVM and KNN can take a minute or more."
              onRetry={() => loadLearning(true)}
            >
              {(curve) => <LearningCurveChart data={curve} />}
            </Pending>
            <p className="mt-auto rounded-2xl bg-muted/60 px-4 py-3 text-sm leading-relaxed text-muted-foreground">
              <span className="font-medium text-foreground">How to read it:</span> a wide gap
              between the two lines means the model memorises the training data (overfitting).
              Both lines levelling off at a low score means the model is too simple
              (underfitting).
            </p>
          </Panel>
        </Reveal>

        <Reveal index={4} className="@3xl:col-span-2">
          <Panel
            className="h-full"
            title="Feature importance"
            subtitle={
              importance.data?.method === "impurity"
                ? "Top 15 features by how much they reduce impurity in the trees."
                : "Top 15 features by how much accuracy drops when each one is shuffled (± 1 standard deviation)."
            }
            actions={
              importance.data && importance.status === "loading" && <Working>Updating…</Working>
            }
          >
            <Pending
              resource={importance}
              height={486}
              waiting="Measuring how much the model relies on each feature. Quick for most models; SVM and KNN can take a minute or more."
              onRetry={() => loadImportance(true)}
            >
              {(items) => <ImportanceChart data={items} />}
            </Pending>
          </Panel>
        </Reveal>
      </div>
    </div>
  )
}
