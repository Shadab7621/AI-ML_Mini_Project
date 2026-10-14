import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react"

import { FoldChart } from "@/components/charts/model-charts"
import { Panel, Pending, Reveal, Stale, StatTile, Working } from "@/components/common"
import { NeedsModel } from "@/components/needs-model"
import { ModelSummary, PageHeader } from "@/components/page-header"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import type { CrossValidation } from "@/lib/api"
import { formatPct, formatScore, statusOf } from "@/lib/format"
import { WhenActive } from "@/state/active"
import { useApp } from "@/state/app-state"
import { cn } from "@/lib/utils"

const DESCRIPTION =
  "The whole dataset is split into 5 parts; the model is trained 5 times, each time tested on a different part. Similar scores mean the result is stable."

function Delta({ value }: { value: number }) {
  const flat = Math.abs(value) < 5e-7
  const Icon = flat ? Minus : value > 0 ? ArrowUpRight : ArrowDownRight
  return (
    <span
      className={cn(
        "inline-flex items-center justify-end gap-1 font-medium tabular-nums",
        flat ? "text-muted-foreground" : value > 0 ? "text-good-ink" : "text-bad-ink"
      )}
    >
      <Icon className="size-4" aria-hidden />
      <span className="sr-only">{flat ? "Same as" : value > 0 ? "Above" : "Below"} the average: </span>
      {flat ? "0.00" : `${value > 0 ? "+" : "−"}${Math.abs(value * 100).toFixed(2)}`} pts
    </span>
  )
}

function FoldTable({ data }: { data: CrossValidation }) {
  return (
    <Table className="text-[15px]">
      <TableHeader>
        <TableRow className="hover:bg-transparent">
          <TableHead className="px-3 text-muted-foreground">Fold</TableHead>
          <TableHead className="px-3 text-right text-muted-foreground">Accuracy</TableHead>
          <TableHead className="px-3 text-right text-muted-foreground">From average</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {data.scores.map((score, i) => (
          <TableRow key={i}>
            <TableCell className="px-3 py-3 font-medium">Fold {i + 1}</TableCell>
            <TableCell className="px-3 text-right tabular-nums">{formatScore(score)}</TableCell>
            <TableCell className="px-3 text-right">
              <Delta value={score - data.mean} />
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

export function CrossValidationPage() {
  const { result, cv, loadCv } = useApp()


  const data = result.data
  if (!data) {
    return (
      <div className="flex flex-col gap-5">
        <PageHeader title="Cross-validation" description={DESCRIPTION} />
        <NeedsModel
          what="its cross-validation scores"
          skeleton={<Skeleton className="h-[420px] rounded-[24px]" />}
        />
      </div>
    )
  }

  const scores = cv.data
  const updating = cv.status !== "ready"

  return (
    <div className="flex flex-col gap-5">
      <WhenActive load={loadCv} />
      <PageHeader title="Cross-validation" description={DESCRIPTION} />
      <ModelSummary result={data} updating={result.status === "loading"} />

      <Reveal index={0}>
        <Stale when={Boolean(scores) && updating} className="grid grid-cols-1 gap-4 @sm:grid-cols-2 @3xl:grid-cols-4">
        <StatTile
          label="Average accuracy"
          value={scores ? formatPct(scores.mean, 2) : "…"}
          status={scores ? statusOf(scores.mean) : undefined}
          hint="across the 5 folds"
        />
        <StatTile
          label="Standard deviation"
          value={scores ? `${(scores.std * 100).toFixed(2)} pts` : "…"}
          hint="smaller means more stable"
        />
        <StatTile
          label="Lowest fold"
          value={scores ? formatPct(scores.min, 2) : "…"}
          hint="the hardest split"
        />
        <StatTile
          label="Highest fold"
          value={scores ? formatPct(scores.max, 2) : "…"}
          hint="the easiest split"
        />
        </Stale>
      </Reveal>

      <Reveal index={1} className="grid grid-cols-1 gap-5 @5xl:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <Panel
          title="Fold by fold"
          subtitle="Each dot is one fold. The closer the dots sit to the dashed average, the more you can trust the score."
          actions={scores && cv.status === "loading" && <Working>Updating…</Working>}
        >
          <Pending
            resource={cv}
            height={356}
            waiting="Training and testing the model 5 times…"
            onRetry={() => loadCv(true)}
          >
            {(folds) => <FoldChart data={folds} />}
          </Pending>
        </Panel>
        <Panel title="Scores" subtitle="Accuracy of every fold and how far it is from the average.">
          <Pending
            resource={cv}
            height={280}
            waiting="Waiting for the 5 folds…"
            onRetry={() => loadCv(true)}
          >
            {(folds) => <FoldTable data={folds} />}
          </Pending>
        </Panel>
      </Reveal>
    </div>
  )
}
