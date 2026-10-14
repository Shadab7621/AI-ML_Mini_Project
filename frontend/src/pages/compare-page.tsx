import * as React from "react"
import { Download, Trophy } from "lucide-react"

import { LeaderboardChart } from "@/components/charts/model-charts"
import {
  EmptyState,
  ErrorNote,
  Panel,
  Reveal,
  Segmented,
  Stale,
  Working,
} from "@/components/common"
import { GlassButton } from "@/components/glass"
import { PageHeader } from "@/components/page-header"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import type { Comparison, MetricKey } from "@/lib/api"
import { formatScore } from "@/lib/format"
import { METRIC_ORDER, useActions } from "@/state/actions"
import { useApp } from "@/state/app-state"
import { cn } from "@/lib/utils"

const DESCRIPTION =
  "All 7 models trained with their default settings on the same training set and scored on the same test set."

const RANK_BY: { value: MetricKey; label: string }[] = [
  { value: "Accuracy", label: "Accuracy" },
  { value: "Precision", label: "Precision" },
  { value: "Recall", label: "Recall" },
  { value: "F1 Score", label: "F1 score" },
  { value: "ROC-AUC", label: "ROC-AUC" },
]

const HEADINGS: Record<MetricKey, string> = {
  Accuracy: "Accuracy",
  Precision: "Precision",
  Recall: "Recall",
  "F1 Score": "F1",
  MCC: "MCC (0–1)",
  "Cohen κ": "Kappa (0–1)",
  "ROC-AUC": "ROC-AUC",
  "Avg Precision": "Avg precision",
}

const same = (a: number | null, b: number | null) =>
  a !== null && b !== null && Math.abs(a - b) < 1e-12

function BestModel({ data }: { data: Comparison }) {
  const best = data.rows.find((row) => row.model === data.best)
  if (!best) return null
  const tied = data.rows.filter(
    (row) => row.model !== best.model && same(row.metrics["F1 Score"], best.metrics["F1 Score"])
  )
  const auc = best.metrics["ROC-AUC"]
  return (
    <div className="squircle flex flex-wrap items-center gap-x-5 gap-y-4 rounded-[24px] bg-card p-5 ring-1 ring-border sm:p-6">
      <span className="grid size-12 shrink-0 place-items-center rounded-full bg-foreground text-background">
        <Trophy className="size-6" aria-hidden />
      </span>
      <div className="grid min-w-0 flex-1 gap-1">
        <p className="text-sm text-muted-foreground">Best model by F1 score</p>
        <p className="text-2xl font-semibold tracking-[-0.02em]">
          {best.model}
          {tied.length > 0 && (
            <span className="ml-2 text-base font-normal text-muted-foreground">
              tied with {tied.map((row) => row.model).join(", ")}
            </span>
          )}
        </p>
      </div>
      <dl className="flex flex-wrap gap-x-8 gap-y-2">
        {[
          ["F1", formatScore(best.metrics["F1 Score"])],
          ["Accuracy", formatScore(best.metrics.Accuracy)],
          ["ROC-AUC", auc === null ? "n/a" : formatScore(auc)],
        ].map(([label, value]) => (
          <div key={label}>
            <dt className="text-[13px] text-muted-foreground">{label}</dt>
            <dd className="text-xl font-semibold tabular-nums">{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

function Leaderboard({ data, rankBy }: { data: Comparison; rankBy: MetricKey }) {
  const rows = [...data.rows].sort(
    (a, b) => (b.metrics[rankBy] ?? -1) - (a.metrics[rankBy] ?? -1)
  )
  const top = Object.fromEntries(
    METRIC_ORDER.map((key) => [key, Math.max(...data.rows.map((row) => row.metrics[key] ?? -1))])
  )

  return (
    <Table className="text-[15px]">
      <TableHeader>
        <TableRow className="hover:bg-transparent">
          <TableHead className="w-10 px-3 text-muted-foreground">#</TableHead>
          <TableHead className="px-3 text-muted-foreground">Model</TableHead>
          {METRIC_ORDER.map((key) => (
            <TableHead
              key={key}
              className={cn(
                "px-3 text-right",
                key === rankBy ? "text-foreground" : "text-muted-foreground"
              )}
            >
              {HEADINGS[key]}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map((row, i) => (
          <TableRow key={row.model}>
            <TableCell className="px-3 text-muted-foreground tabular-nums">{i + 1}</TableCell>
            <TableCell className="px-3 py-3 font-medium">{row.model}</TableCell>
            {METRIC_ORDER.map((key) => {
              const value = row.metrics[key]
              const best = same(value, top[key])
              return (
                <TableCell
                  key={key}
                  className={cn(
                    "px-3 text-right tabular-nums",
                    best ? "font-semibold" : "text-muted-foreground",
                    key === rankBy && "bg-muted/50"
                  )}
                >
                  {value === null ? "n/a" : formatScore(value)}
                </TableCell>
              )
            })}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

export function ComparePage() {
  const { comparison } = useApp()
  const { compareAndShow, exportComparison } = useActions()
  const [rankBy, setRankBy] = React.useState<MetricKey>("F1 Score")
  const data = comparison.data
  const chartRows = React.useMemo(
    () => (data?.rows ?? []).map((row) => ({ model: row.model, value: row.metrics[rankBy] ?? 0 })),
    [data, rankBy]
  )

  if (!data) {
    return (
      <div className="flex flex-col gap-5">
        <PageHeader title="Compare" description={DESCRIPTION} />
        {comparison.status === "loading" ? (
          <div className="flex flex-col gap-5">
            <Working>Training and scoring all 7 models. This takes a few seconds…</Working>
            <Skeleton className="h-[108px] rounded-[24px]" />
            <Skeleton className="h-[420px] rounded-[24px]" />
          </div>
        ) : (
          <>
            {comparison.status === "error" && (
              <ErrorNote
                message={comparison.error ?? "The comparison failed."}
                onRetry={compareAndShow}
              />
            )}
            <EmptyState
              icon={Trophy}
              title="Which model is best?"
              action={
                <GlassButton tone="prominent" size="lg" onClick={compareAndShow}>
                  <Trophy aria-hidden />
                  Compare all 7 models
                </GlassButton>
              }
            >
              Train every model on the same data and see them ranked side by side. The result
              is remembered, so coming back here is instant.
            </EmptyState>
          </>
        )}
      </div>
    )
  }

  const metricLabel = RANK_BY.find((option) => option.value === rankBy)?.label ?? rankBy

  return (
    <div className="flex flex-col gap-5">
      <PageHeader title="Compare" description={DESCRIPTION}>
        <GlassButton onClick={exportComparison}>
          <Download aria-hidden />
          Download CSV
        </GlassButton>
      </PageHeader>

      <Stale when={comparison.status !== "ready"} className="flex flex-col gap-5">
        <Reveal index={0}>
          <BestModel data={data} />
        </Reveal>

        <Reveal index={1} className="flex flex-wrap items-center gap-3">
          <span className="text-[15px] font-medium">Rank by</span>
          <Segmented
            label="Rank models by"
            value={rankBy}
            options={RANK_BY}
            onChange={setRankBy}
            className="w-full max-w-xl bg-card ring-1 ring-border"
          />
        </Reveal>

        <Reveal index={2} className="grid grid-cols-1 gap-5 @7xl:grid-cols-[minmax(0,1fr)_minmax(0,1.35fr)]">
          <Panel title={metricLabel} subtitle="Every model on the test set, best first.">
            <LeaderboardChart metric={metricLabel} rows={chartRows} />
          </Panel>
          <Panel
            title="Leaderboard"
            subtitle="All scores. The best value in each column is in bold."
          >
            <Leaderboard data={data} rankBy={rankBy} />
          </Panel>
        </Reveal>
      </Stale>
    </div>
  )
}
