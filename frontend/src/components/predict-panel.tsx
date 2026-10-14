// Try the trained model on a single example: type the values in, or load a row
// from the test set, and see what the model says.
import * as React from "react"
import { Dices, Wand2 } from "lucide-react"

import { ErrorNote, Panel, StatusChip, Working } from "@/components/common"
import { GlassButton } from "@/components/glass"
import {
  api,
  type ExampleValues,
  type Field,
  type ModelRequest,
  type Prediction,
  type Problem,
} from "@/lib/api"
import { featureLabel, valueLabel } from "@/lib/features"
import { capitalise, formatPct } from "@/lib/format"
import { cn } from "@/lib/utils"

const CONTROL =
  "h-10 w-full min-w-0 rounded-xl border border-input bg-transparent px-3 text-[15px] outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30"

const defaultsOf = (fields: Field[]): ExampleValues =>
  Object.fromEntries(fields.map((field) => [field.name, field.default]))

function FieldInput({
  field,
  value,
  dataset,
  onChange,
}: {
  field: Field
  value: number | string | null
  dataset: string
  onChange: (value: number | string | null) => void
}) {
  const id = React.useId()
  return (
    <div className="grid min-w-0 gap-1.5">
      <label htmlFor={id} className="truncate text-[13px] text-muted-foreground">
        {featureLabel(field.name)}
      </label>
      {field.kind === "category" ? (
        <select
          id={id}
          className={CONTROL}
          value={value === null ? "" : String(value)}
          onChange={(event) => onChange(event.target.value === "" ? null : event.target.value)}
        >
          {field.options.map((option) => (
            <option key={option} value={option}>
              {valueLabel(dataset, field.name, option)}
            </option>
          ))}
          <option value="">unknown</option>
        </select>
      ) : (
        <input
          id={id}
          type="number"
          inputMode="decimal"
          step={field.whole ? 1 : "any"}
          placeholder="unknown"
          title={`In the data this runs from ${field.min} to ${field.max}.`}
          className={cn(CONTROL, "tabular-nums")}
          value={value === null ? "" : value}
          onChange={(event) =>
            onChange(event.target.value === "" ? null : Number(event.target.value))
          }
        />
      )}
    </div>
  )
}

type Answer =
  | { key: string; status: "loading" }
  | { key: string; status: "ready"; data: Prediction }
  | { key: string; status: "error"; error: string }

export function PredictPanel({ problem, request }: { problem: Problem; request: ModelRequest }) {
  const [values, setValues] = React.useState<ExampleValues>(() => defaultsOf(problem.fields))
  /** The true outcome, known only while the form holds an untouched row of the test set. */
  const [truth, setTruth] = React.useState<string | null>(null)
  const [answer, setAnswer] = React.useState<Answer | null>(null)
  const [loadingRow, setLoadingRow] = React.useState(false)
  const formId = React.useId()

  // An answer belongs to one model and one set of values; change either and it no longer applies.
  const keyOf = (example: ExampleValues) => JSON.stringify([request, example])
  const current = answer?.key === keyOf(values) ? answer : null
  const positive = problem.class_names[1]

  const predict = (example: ExampleValues) => {
    const key = keyOf(example)
    setAnswer({ key, status: "loading" })
    api.predict(request, example).then(
      (data) => setAnswer((now) => (now?.key === key ? { key, status: "ready", data } : now)),
      (error: unknown) =>
        setAnswer((now) =>
          now?.key === key
            ? { key, status: "error", error: error instanceof Error ? error.message : "Prediction failed." }
            : now
        )
    )
  }

  const randomRow = () => {
    setLoadingRow(true)
    api.example(request).then(
      (example) => {
        setLoadingRow(false)
        setValues(example.values)
        setTruth(example.truth)
        predict(example.values)
      },
      () => setLoadingRow(false)
    )
  }

  const edit = (name: string, value: number | string | null) => {
    setValues((now) => ({ ...now, [name]: value }))
    setTruth(null)
  }

  return (
    <Panel
      title="Try the model on one example"
      subtitle="Fill in the values, or load a row from the test set, and see what the trained model says."
      actions={
        <div className="flex flex-wrap gap-2.5">
          <GlassButton loading={loadingRow} onClick={randomRow}>
            {!loadingRow && <Dices aria-hidden />}
            Random test row
          </GlassButton>
          {/* submits the form below, so pressing Enter in a field does the same */}
          <GlassButton tone="prominent" type="submit" form={formId}>
            <Wand2 aria-hidden />
            Predict
          </GlassButton>
        </div>
      }
    >
      <div className="grid grid-cols-1 gap-5 @4xl:grid-cols-[minmax(0,1fr)_minmax(0,19rem)]">
        <form
          id={formId}
          className="grid max-h-[344px] grid-cols-2 content-start gap-x-4 gap-y-3.5 overflow-y-auto pr-1 [scrollbar-width:thin] @2xl:grid-cols-3 @5xl:grid-cols-4"
          onSubmit={(event) => {
            event.preventDefault()
            predict(values)
          }}
        >
          {problem.fields.map((field) => (
            <FieldInput
              key={field.name}
              field={field}
              dataset={problem.dataset.id}
              value={values[field.name] ?? null}
              onChange={(value) => edit(field.name, value)}
            />
          ))}
        </form>

        <div
          aria-live="polite"
          className="flex min-h-[168px] flex-col justify-center gap-3 rounded-2xl bg-muted/60 p-5"
        >
          {!current && (
            <p className="text-[15px] text-muted-foreground">
              Press <span className="font-medium text-foreground">Predict</span> to see the
              model's answer for these values.
            </p>
          )}
          {current?.status === "loading" && <Working>Asking the model…</Working>}
          {current?.status === "error" && <ErrorNote message={current.error} />}
          {current?.status === "ready" && (
            <>
              <div className="grid gap-1">
                <p className="text-sm text-muted-foreground">The model says</p>
                <p className="text-[28px] leading-tight font-semibold tracking-[-0.02em]">
                  {capitalise(current.data.label)}
                </p>
              </div>
              {current.data.probability !== null && (
                <div className="grid gap-1.5">
                  <div className="h-2 overflow-hidden rounded-full bg-background" aria-hidden>
                    <div
                      className="h-full rounded-full bg-foreground transition-[width] duration-500"
                      style={{ width: `${current.data.probability * 100}%` }}
                    />
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Its estimate that this is {positive}:{" "}
                    <span className="font-semibold text-foreground tabular-nums">
                      {formatPct(current.data.probability)}
                    </span>
                  </p>
                </div>
              )}
              {truth !== null && (
                <StatusChip status={truth === current.data.label ? "good" : "bad"}>
                  {truth === current.data.label
                    ? `Correct: this test row is labelled ${truth}`
                    : `Wrong: this test row is labelled ${truth}`}
                </StatusChip>
              )}
            </>
          )}
        </div>
      </div>
    </Panel>
  )
}
