import { Check, CircleCheck, TriangleAlert } from "lucide-react"

import { formatInt, formatPct } from "@/lib/format"
import { cn } from "@/lib/utils"

// [actual][predicted]; index 1 is the outcome being detected
const MEANING = [
  ["Correct", "False alarm"],
  ["Missed", "Correct"],
]

/** 2 × 2 grid: rows are what a test row really is, columns what the model said. */
export function ConfusionMatrix({
  matrix,
  classNames,
}: {
  matrix: [[number, number], [number, number]]
  classNames: [string, string]
}) {
  return (
    <div
      role="table"
      aria-label="Confusion matrix"
      className="grid grid-cols-[auto_minmax(0,1fr)_minmax(0,1fr)] gap-2"
    >
      <div role="row" className="contents">
        <span role="columnheader" />
        {classNames.map((name) => (
          <span
            key={name}
            role="columnheader"
            className="pb-1 text-center text-[13px] leading-tight text-muted-foreground"
          >
            Predicted <span className="font-medium text-foreground">{name}</span>
          </span>
        ))}
      </div>

      {matrix.map((row, actual) => {
        const total = row[0] + row[1]
        return (
          <div key={actual} role="row" className="contents">
            <span
              role="rowheader"
              className="flex w-[72px] flex-col justify-center pr-1 text-[13px] leading-tight text-muted-foreground"
            >
              Actually
              <span className="font-medium break-words text-foreground">{classNames[actual]}</span>
            </span>
            {row.map((count, predicted) => {
              const correct = actual === predicted
              const share = total ? count / total : 0
              const wrong = !correct && count > 0
              const Icon = correct ? CircleCheck : wrong ? TriangleAlert : Check
              return (
                <div
                  key={predicted}
                  role="cell"
                  className={cn(
                    "flex min-h-[122px] min-w-0 flex-col justify-between rounded-2xl p-3.5",
                    wrong && "bg-bad/12 ring-1 ring-bad/35",
                    !correct && !wrong && "bg-muted/60"
                  )}
                  style={
                    correct
                      ? {
                          background: `color-mix(in oklab, var(--foreground) ${Math.round(5 + share * 13)}%, transparent)`,
                        }
                      : undefined
                  }
                >
                  <span
                    className={cn(
                      "flex items-center gap-1.5 text-[13px] font-medium",
                      correct ? "text-good-ink" : wrong ? "text-bad-ink" : "text-muted-foreground"
                    )}
                  >
                    <Icon className="size-4 shrink-0" aria-hidden />
                    {wrong || correct ? MEANING[actual][predicted] : "None"}
                  </span>
                  <span>
                    <span className="block text-[28px] leading-none font-semibold tracking-[-0.02em]">
                      {formatInt(count)}
                    </span>
                    <span className="mt-1.5 block text-[13px] leading-tight text-muted-foreground">
                      {formatPct(share)} of {classNames[actual]}
                    </span>
                  </span>
                </div>
              )
            })}
          </div>
        )
      })}
    </div>
  )
}
