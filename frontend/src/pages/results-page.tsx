import { Download } from "lucide-react"

import { ResultsBento } from "@/components/bento"
import { Panel, Reveal, Stale, StatTile } from "@/components/common"
import { GlassButton } from "@/components/glass"
import { NeedsModel } from "@/components/needs-model"
import { ModelSummary, PageHeader } from "@/components/page-header"
import { PredictPanel } from "@/components/predict-panel"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import type { ReportRow, TrainResult } from "@/lib/api"
import { capitalise, formatInt, formatPct, formatScore, statusOf } from "@/lib/format"
import { useActions } from "@/state/actions"
import { WhenActive } from "@/state/active"
import { useApp } from "@/state/app-state"

const DESCRIPTION = "How the trained model did on the 30% of the data it never saw."

function ReportTable({ result }: { result: TrainResult }) {
  const { report } = result
  const row = (key: string) => report[key] as ReportRow
  const classes = [
    [result.class_names[0], row("negative")],
    [result.class_names[1], row("positive")],
  ] as const
  const averages = [
    ["Macro average", row("macro avg")],
    ["Weighted average", row("weighted avg")],
  ] as const
  const accuracy = report.accuracy as number
  const number = "px-3 text-right tabular-nums"

  return (
    <Table className="text-[15px]">
      <TableHeader>
        <TableRow className="hover:bg-transparent">
          <TableHead className="px-3 text-muted-foreground">Class</TableHead>
          {["Precision", "Recall", "F1 score", "Support"].map((heading) => (
            <TableHead key={heading} className="px-3 text-right text-muted-foreground">
              {heading}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {classes.map(([name, scores]) => (
          <TableRow key={name}>
            <TableCell className="px-3 py-3 font-medium">{capitalise(name)}</TableCell>
            <TableCell className={number}>{formatScore(scores.precision)}</TableCell>
            <TableCell className={number}>{formatScore(scores.recall)}</TableCell>
            <TableCell className={number}>{formatScore(scores["f1-score"])}</TableCell>
            <TableCell className={number}>{formatInt(scores.support)}</TableCell>
          </TableRow>
        ))}
        <TableRow>
          <TableCell className="px-3 py-3 text-muted-foreground">Accuracy</TableCell>
          <TableCell className={number} />
          <TableCell className={number} />
          <TableCell className={number}>{formatScore(accuracy)}</TableCell>
          <TableCell className={number}>{formatInt(result.y_true.length)}</TableCell>
        </TableRow>
        {averages.map(([name, scores]) => (
          <TableRow key={name}>
            <TableCell className="px-3 py-3 text-muted-foreground">{name}</TableCell>
            <TableCell className={number}>{formatScore(scores.precision)}</TableCell>
            <TableCell className={number}>{formatScore(scores.recall)}</TableCell>
            <TableCell className={number}>{formatScore(scores["f1-score"])}</TableCell>
            <TableCell className={number}>{formatInt(scores.support)}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

function ResultsSkeleton() {
  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-1 gap-4 @3xl:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
        <Skeleton className="h-[300px] rounded-[32px] @3xl:row-span-2" />
        <Skeleton className="h-[142px] rounded-[28px]" />
        <Skeleton className="h-[142px] rounded-[28px]" />
      </div>
      <div className="grid grid-cols-1 gap-4 @sm:grid-cols-2 @2xl:grid-cols-3 @5xl:grid-cols-5">
        {[0, 1, 2, 3, 4].map((i) => (
          <Skeleton key={i} className="h-[132px] rounded-[22px]" />
        ))}
      </div>
    </div>
  )
}

export function ResultsPage() {
  const { result, cv, loadCv, dataset, trained } = useApp()
  const { exportPredictions } = useActions()


  const data = result.data
  if (!data) {
    return (
      <div className="flex flex-col gap-5">
        <PageHeader title="Results" description={DESCRIPTION} />
        <NeedsModel what="its scores" skeleton={<ResultsSkeleton />} />
      </div>
    )
  }

  const { metrics } = data
  const problem = dataset.data
  const unit = problem?.dataset.unit ?? "rows"
  const wrong = data.y_true.filter((truth, i) => truth !== data.y_pred[i]).length
  const average = metrics["Avg Precision"]
  const fresh = result.status === "ready"

  return (
    <div className="flex flex-col gap-5">
      <WhenActive load={loadCv} />
      <PageHeader title="Results" description={DESCRIPTION} />
      <ModelSummary result={data} updating={result.status === "loading"} />

      <Stale when={!fresh} className="flex flex-col gap-5">
        <Reveal index={0}>
          <ResultsBento result={data} cv={cv} problem={problem} />
        </Reveal>

        <Reveal index={1} className="grid grid-cols-1 gap-4 @sm:grid-cols-2 @2xl:grid-cols-3 @5xl:grid-cols-5">
          <StatTile
            label="Precision"
            value={formatPct(metrics.Precision)}
            status={statusOf(metrics.Precision)}
            hint="right when it names a class"
          />
          <StatTile
            label="Recall"
            value={formatPct(metrics.Recall)}
            status={statusOf(metrics.Recall)}
            hint="of each class found"
          />
          <StatTile
            label="Average precision"
            value={average === null ? "n/a" : formatPct(average)}
            status={average === null ? undefined : statusOf(average)}
            hint="area under the PR curve"
          />
          <StatTile
            label="MCC"
            value={formatPct(metrics.MCC)}
            status={statusOf(metrics.MCC)}
            hint="rescaled to 0–1"
          />
          <StatTile
            label="Cohen's kappa"
            value={formatPct(metrics["Cohen κ"])}
            status={statusOf(metrics["Cohen κ"])}
            hint="rescaled to 0–1"
          />
        </Reveal>

        {problem && trained && (
          <Reveal index={2}>
            <PredictPanel
              // a new dataset, target or outcome means different fields: start the form over
              key={`${trained.dataset}|${trained.target}|${trained.positive}`}
              problem={problem}
              request={trained}
            />
          </Reveal>
        )}

        <Reveal index={3} className="grid grid-cols-1 gap-5 @5xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
          <Panel
            title="Classification report"
            subtitle={`Scores for each class. Support is the number of test ${unit} of that class.`}
          >
            <ReportTable result={data} />
          </Panel>
          <Panel
            title="Export predictions"
            subtitle="Every test row with its true label and the model's answer."
          >
            <dl className="grid grid-cols-2 gap-3">
              <div className="rounded-2xl bg-muted/60 p-4">
                <dt className="text-sm text-muted-foreground">Predictions</dt>
                <dd className="text-2xl font-semibold tracking-[-0.02em]">
                  {formatInt(data.y_true.length)}
                </dd>
              </div>
              <div className="rounded-2xl bg-muted/60 p-4">
                <dt className="text-sm text-muted-foreground">Wrong</dt>
                <dd className="text-2xl font-semibold tracking-[-0.02em]">{formatInt(wrong)}</dd>
              </div>
            </dl>
            <GlassButton className="mt-auto w-full" onClick={exportPredictions}>
              <Download aria-hidden />
              Download predictions CSV
            </GlassButton>
          </Panel>
        </Reveal>
      </Stale>
    </div>
  )
}
