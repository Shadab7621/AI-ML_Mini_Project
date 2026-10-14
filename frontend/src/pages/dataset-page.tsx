import {
  ChevronRight,
  Download,
  FlaskConical,
  Layers,
  ListChecks,
  Play,
  Rows3,
  Trophy,
  type LucideIcon,
} from "lucide-react"

import {
  AssociationHeatmap,
  ClassDonut,
  DistributionGrid,
  TargetLinkList,
} from "@/components/charts/dataset-charts"
import { ErrorNote, Panel, Reveal, Stale, StatTile } from "@/components/common"
import { ColumnsPanel, DataPanel } from "@/components/data-panel"
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
import { valueLabel } from "@/lib/features"
import { formatInt, formatPct } from "@/lib/format"
import { useActions } from "@/state/actions"
import { useApp } from "@/state/app-state"
import { cn } from "@/lib/utils"

const DESCRIPTION = "The data every model is trained and tested on."

function QuickAction({
  icon: Icon,
  title,
  description,
  onClick,
}: {
  icon: LucideIcon
  title: string
  description: string
  onClick?: () => void
}) {
  return (
    <li>
      <button
        type="button"
        disabled={!onClick}
        onClick={onClick}
        className={cn(
          "group flex w-full items-center gap-4 rounded-2xl px-3 py-3 text-left outline-none transition-colors focus-visible:ring-3 focus-visible:ring-ring",
          onClick ? "hover:bg-muted" : "opacity-50"
        )}
      >
        <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-muted ring-1 ring-border transition-colors group-hover:bg-card">
          <Icon className="size-5" aria-hidden />
        </span>
        <span className="grid min-w-0 flex-1 gap-0.5">
          <span className="text-[15px] font-semibold">{title}</span>
          <span className="text-sm text-muted-foreground">{description}</span>
        </span>
        <ChevronRight
          className="size-5 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5"
          aria-hidden
        />
      </button>
    </li>
  )
}

export function DatasetPage() {
  const { dataset, meta, reload, classifier, canTrain } = useApp()
  const { trainAndShow, compareAndShow, exportPredictions, exportComparison } = useActions()
  const data = dataset.data

  if (!data) {
    return (
      <div className="flex flex-col gap-5">
        <PageHeader title="Dataset" description={DESCRIPTION} />
        {dataset.status === "error" ? (
          // when the server itself is down, the note above the page already says so
          meta.status !== "error" && (
            <ErrorNote message={dataset.error ?? "Could not load the dataset."} onRetry={reload} />
          )
        ) : (
          <>
            <Skeleton className="h-[230px] rounded-[24px]" />
            <div className="grid grid-cols-1 gap-4 @sm:grid-cols-2 @3xl:grid-cols-4">
              {[0, 1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-[124px] rounded-[22px]" />
              ))}
            </div>
            <div className="grid grid-cols-1 gap-5 @4xl:grid-cols-2">
              <Skeleton className="h-[300px] rounded-[24px]" />
              <Skeleton className="h-[300px] rounded-[24px]" />
            </div>
          </>
        )}
      </div>
    )
  }

  const { unit } = data.dataset
  const [negative, positive] = data.class_names
  const kinds = [
    data.n_numeric ? `${data.n_numeric} ${data.n_numeric === 1 ? "number" : "numbers"}` : null,
    data.n_categorical
      ? `${data.n_categorical} ${data.n_categorical === 1 ? "category" : "categories"}`
      : null,
  ].filter(Boolean)
  const mushrooms = data.dataset.id === "mushrooms"

  return (
    <div className="flex flex-col gap-5">
      <PageHeader title="Dataset" description={DESCRIPTION} />

      <Reveal index={0}>
        <DataPanel problem={data} />
      </Reveal>

      {/* while another dataset loads, the previous one stays on screen, dimmed */}
      <Stale when={dataset.status !== "ready"} className="flex flex-col gap-5">
        <Reveal index={1} className="grid grid-cols-1 gap-4 @sm:grid-cols-2 @3xl:grid-cols-4">
          <StatTile label="Total samples" value={formatInt(data.n_samples)} hint={unit} icon={Rows3} />
          <StatTile
            label="Features"
            value={data.n_features}
            hint={kinds.join(", ")}
            icon={Layers}
          />
          <StatTile
            label="Training set"
            value={formatInt(data.n_train)}
            hint={`${formatPct(data.n_train / data.n_samples, 0)} of the data`}
            icon={FlaskConical}
          />
          <StatTile
            label="Test set"
            value={formatInt(data.n_test)}
            hint={`${formatPct(data.n_test / data.n_samples, 0)}, never seen in training`}
            icon={ListChecks}
          />
        </Reveal>

        <Reveal index={2} className="grid grid-cols-1 gap-5 @4xl:grid-cols-2">
          <Panel
            title="Class balance"
            subtitle={`How the ${unit} split between ${negative} and ${positive}.`}
          >
            <div className="grid flex-1 place-items-center py-2">
              <ClassDonut counts={data.class_counts} unit={unit} />
            </div>
          </Panel>
          <Panel title="Quick actions" subtitle="Shortcuts to the common steps.">
            <ul className="-mx-3 grid gap-1">
              <QuickAction
                icon={Play}
                title={classifier ? `Train ${classifier.short}` : "Train the selected model"}
                description="Fit it on the training set and score it on the test set."
                onClick={canTrain ? trainAndShow : undefined}
              />
              <QuickAction
                icon={Trophy}
                title="Compare all 7 models"
                description="Train every model with default settings, side by side."
                onClick={canTrain ? compareAndShow : undefined}
              />
              <QuickAction
                icon={Download}
                title="Export predictions"
                description={
                  exportPredictions ? "CSV of every test-set prediction." : "Train a model first."
                }
                onClick={exportPredictions}
              />
              <QuickAction
                icon={Download}
                title="Export comparison"
                description={exportComparison ? "CSV of the leaderboard." : "Run the comparison first."}
                onClick={exportComparison}
              />
            </ul>
          </Panel>
        </Reveal>

        <Reveal index={3} className="grid grid-cols-1 gap-5 @5xl:grid-cols-[auto_minmax(0,1fr)]">
          <Panel
            title="How the columns relate"
            subtitle="How strongly each pair of columns moves together. Hover a square for the value."
          >
            <AssociationHeatmap association={data.association} />
          </Panel>
          <Panel
            title="Strongest links to the target"
            subtitle={`The columns that say the most about "${data.target}". 0 means no link, 1 a perfect one.`}
          >
            <TargetLinkList features={data.top_features} />
          </Panel>
        </Reveal>

        <Reveal index={4}>
          <Panel
            title="Top features by outcome"
            subtitle={`For those columns: how many ${unit} of each outcome have each value.`}
          >
            <DistributionGrid
              features={data.top_features}
              classNames={data.class_names}
              dataset={data.dataset.id}
            />
          </Panel>
        </Reveal>

        <Reveal index={5}>
          <ColumnsPanel problem={data} />
        </Reveal>

        <Reveal index={6}>
          <Panel
            title="Data preview"
            subtitle={
              `The first ${data.preview.rows.length} rows of the file.` +
              (mushrooms
                ? " Every value is a one-letter code; hover a cell to see what it means."
                : " A dash means the value is missing.")
            }
          >
            <Table className="text-[15px]">
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  {data.preview.columns.map((column) => (
                    <TableHead
                      key={column}
                      className={cn(
                        "px-3 text-[13px]",
                        column === data.target ? "text-foreground" : "text-muted-foreground"
                      )}
                    >
                      {column}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.preview.rows.map((row, i) => (
                  <TableRow key={i}>
                    {row.map((cell, j) => {
                      const column = data.preview.columns[j]
                      const text = cell === null ? "" : String(cell)
                      return (
                        <TableCell
                          key={j}
                          title={mushrooms ? valueLabel(data.dataset.id, column, text) : undefined}
                          className={cn(
                            "max-w-64 truncate px-3 py-2.5",
                            mushrooms && "font-mono",
                            column === data.target && "font-semibold",
                            cell === null && "text-muted-foreground"
                          )}
                        >
                          {cell === null ? "–" : text}
                        </TableCell>
                      )
                    })}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Panel>
        </Reveal>
      </Stale>
    </div>
  )
}
