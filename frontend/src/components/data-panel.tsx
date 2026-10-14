// Choosing what to work on: the dataset, the column to predict and the outcome to
// detect, plus uploading a CSV file. Also the table of how each column is used.
import * as React from "react"
import { CircleAlert, Trash2, Upload, X } from "lucide-react"

import { Panel, Working } from "@/components/common"
import { GlassButton } from "@/components/glass"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import type { ColumnProfile, Problem } from "@/lib/api"
import { formatInt, formatPct } from "@/lib/format"
import { useApp } from "@/state/app-state"
import { cn } from "@/lib/utils"

function Choice({
  label,
  hint,
  children,
}: {
  label: string
  hint: React.ReactNode
  children: (labelId: string) => React.ReactNode
}) {
  const id = React.useId()
  return (
    <div className="grid content-start gap-2">
      <span id={id} className="text-[15px] font-medium">
        {label}
      </span>
      {children(id)}
      <p className="text-[13px] leading-snug text-muted-foreground">{hint}</p>
    </div>
  )
}

const TRIGGER = "h-11! w-full rounded-xl px-3.5 text-[15px] font-medium"

export function DataPanel({ problem }: { problem: Problem }) {
  const {
    datasets,
    dataset,
    notice,
    dismissNotice,
    chooseDataset,
    chooseTarget,
    choosePositive,
    uploading,
    uploadDataset,
    removeDataset,
  } = useApp()
  const fileInput = React.useRef<HTMLInputElement>(null)

  const all = datasets.data?.datasets ?? []
  const samples = all.filter((item) => item.source === "sample")
  const uploads = all.filter((item) => item.source === "upload")
  const classes = problem.targets.find((item) => item.name === problem.target)?.classes ?? []
  const switching = dataset.status === "loading"
  const limit = datasets.data?.max_upload_mb ?? 10

  return (
    <Panel
      title="Data"
      subtitle={problem.dataset.description}
      actions={switching && <Working>Loading the data…</Working>}
    >
      <div className="grid grid-cols-1 gap-x-5 gap-y-5 @3xl:grid-cols-3">
        <Choice label="Dataset" hint="A sample, or a CSV file of your own.">
          {(labelId) => (
            <Select value={problem.dataset.id} onValueChange={chooseDataset} disabled={switching}>
              <SelectTrigger aria-labelledby={labelId} className={TRIGGER}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent position="popper" className="rounded-xl">
                <SelectGroup>
                  <SelectLabel className="text-[13px]">Samples</SelectLabel>
                  {samples.map((item) => (
                    <SelectItem key={item.id} value={item.id} className="py-2 text-[15px]">
                      {item.name}
                    </SelectItem>
                  ))}
                </SelectGroup>
                {uploads.length > 0 && (
                  <SelectGroup>
                    <SelectLabel className="text-[13px]">Your files</SelectLabel>
                    {uploads.map((item) => (
                      <SelectItem key={item.id} value={item.id} className="py-2 text-[15px]">
                        {item.name}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                )}
              </SelectContent>
            </Select>
          )}
        </Choice>

        <Choice label="Column to predict" hint="The models learn to guess this column from the others.">
          {(labelId) => (
            <Select value={problem.target} onValueChange={chooseTarget} disabled={switching}>
              <SelectTrigger aria-labelledby={labelId} className={TRIGGER}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent position="popper" className="max-h-80 rounded-xl">
                {problem.targets.map((item) => (
                  <SelectItem key={item.name} value={item.name} className="py-2 text-[15px]">
                    {item.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </Choice>

        <Choice
          label="Outcome to detect"
          hint={
            classes.length > 2
              ? `This column has ${classes.length} values; the others are grouped as "${problem.class_names[0]}".`
              : "The app counts how often the models miss this outcome."
          }
        >
          {(labelId) => (
            <Select value={problem.positive} onValueChange={choosePositive} disabled={switching}>
              <SelectTrigger aria-labelledby={labelId} className={TRIGGER}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent position="popper" className="max-h-80 rounded-xl">
                {classes.map((item) => (
                  <SelectItem key={item.value} value={item.value} className="py-2 text-[15px]">
                    {item.label}
                    <span className="text-muted-foreground">{formatInt(item.count)} rows</span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </Choice>
      </div>

      {notice && (
        <div
          role="alert"
          className="flex items-center gap-3 rounded-2xl bg-bad/10 px-4 py-3 text-[15px] ring-1 ring-bad/30"
        >
          <CircleAlert className="size-5 shrink-0 text-bad-ink" aria-hidden />
          <p className="min-w-0 flex-1">{notice}</p>
          <Button variant="ghost" size="icon" onClick={dismissNotice} aria-label="Dismiss">
            <X aria-hidden />
          </Button>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-x-4 gap-y-3 border-t pt-5">
        <input
          ref={fileInput}
          type="file"
          accept=".csv,.tsv,.txt,.data,text/csv"
          className="sr-only"
          aria-label="CSV file to upload"
          tabIndex={-1}
          onChange={(event) => {
            const file = event.target.files?.[0]
            if (file) uploadDataset(file)
            event.target.value = "" // so the same file can be picked again
          }}
        />
        <GlassButton loading={uploading} onClick={() => fileInput.current?.click()}>
          {!uploading && <Upload aria-hidden />}
          {uploading ? "Uploading…" : "Upload a CSV file"}
        </GlassButton>
        {problem.dataset.source === "upload" && (
          <Button
            variant="ghost"
            className="h-11 rounded-full px-4 text-[15px]"
            onClick={() => removeDataset(problem.dataset.id)}
          >
            <Trash2 aria-hidden />
            Remove this file
          </Button>
        )}
        <p className="min-w-0 flex-1 basis-72 text-[13px] leading-snug text-muted-foreground">
          Up to {limit} MB and at least 50 rows, with column names in the first line. One column
          must hold the classes to predict. The file stays on this computer.
        </p>
      </div>
    </Panel>
  )
}

const KIND: Record<ColumnProfile["kind"], string> = {
  number: "Number",
  category: "Category",
  target: "Predicted",
  ignored: "Not used",
}

/** How the app reads each column of the file. */
export function ColumnsPanel({ problem }: { problem: Problem }) {
  const unused = problem.columns.filter((column) => column.kind === "ignored").length
  return (
    <Panel
      title="Columns"
      subtitle={
        `${problem.n_numeric} used as numbers and ${problem.n_categorical} as categories` +
        (unused ? `; ${unused} not used.` : ".") +
        " Missing numbers are filled with the column's median; a missing category counts as its own value."
      }
    >
      <div className="max-h-[420px] overflow-y-auto [scrollbar-width:thin]">
        <Table className="text-[15px]">
          <TableHeader className="sticky top-0 z-10 bg-card">
            <TableRow className="hover:bg-transparent">
              <TableHead className="px-3 text-muted-foreground">Column</TableHead>
              <TableHead className="px-3 text-muted-foreground">Used as</TableHead>
              <TableHead className="px-3 text-right text-muted-foreground">Different values</TableHead>
              <TableHead className="px-3 text-right text-muted-foreground">Missing</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {problem.columns.map((column) => (
              <TableRow key={column.name}>
                <TableCell className="px-3 py-2.5 font-medium">{column.name}</TableCell>
                <TableCell className="px-3 whitespace-normal">
                  <span
                    className={cn(
                      column.kind === "ignored" && "text-muted-foreground",
                      column.kind === "target" && "font-semibold"
                    )}
                  >
                    {KIND[column.kind]}
                  </span>
                  {column.reason && (
                    <span className="text-muted-foreground">: {column.reason}</span>
                  )}
                </TableCell>
                <TableCell className="px-3 text-right tabular-nums">
                  {formatInt(column.distinct)}
                </TableCell>
                <TableCell className="px-3 text-right tabular-nums">
                  {column.missing === 0 ? (
                    <span className="text-muted-foreground">none</span>
                  ) : (
                    <>
                      {formatInt(column.missing)}{" "}
                      <span className="text-muted-foreground">
                        ({formatPct(column.missing / problem.n_rows, 1)})
                      </span>
                    </>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </Panel>
  )
}
