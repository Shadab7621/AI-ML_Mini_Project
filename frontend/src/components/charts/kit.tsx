/* Shared pieces for every chart.

   The rules the charts follow:
     · black and white first: one series is the ink colour; a second series is
       grey and dashed, so the two differ by form and not only by shade
     · green and red appear only where they mean something: a good or bad score,
       or the two outcomes when one of them is the bad (or the good) one
     · thin marks, hairline solid gridlines, text always in the text colours
     · two or more series get a legend; every chart has a tooltip and a hidden
       data table, so nothing depends on colour or on hovering
*/
import * as React from "react"
import { ResponsiveContainer } from "recharts"

import { cn } from "@/lib/utils"

export const INK = "var(--foreground)"
export const INK_2 = "var(--muted-foreground)"
export const GRID = "var(--grid)"
export const AXIS_TICK = { fill: "var(--muted-foreground)", fontSize: 12 }
export const AXIS_LABEL = { fill: "var(--muted-foreground)", fontSize: 13 }

/** What Recharts hands to a custom tooltip (only the parts used here). */
export interface TipProps {
  active?: boolean
  label?: unknown
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  payload?: readonly any[]
}

export interface Key {
  label: string
  color: string
  dashed?: boolean
  /** "line" for lines, "box" for bars and areas. */
  kind?: "line" | "box"
}

function Swatch({ color, dashed, kind = "line" }: Omit<Key, "label">) {
  if (kind === "box") {
    return <span className="size-2.5 rounded-[3px]" style={{ background: color }} />
  }
  return (
    <svg width="18" height="4" aria-hidden className="shrink-0">
      <line
        x1="1"
        y1="2"
        x2="17"
        y2="2"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeDasharray={dashed ? "4 4" : undefined}
      />
    </svg>
  )
}

export function Legend({ items, className }: { items: Key[]; className?: string }) {
  return (
    <ul className={cn("flex flex-wrap gap-x-5 gap-y-1.5", className)}>
      {items.map((item) => (
        <li key={item.label} className="flex items-center gap-2 text-sm text-muted-foreground">
          <Swatch {...item} />
          {item.label}
        </li>
      ))}
    </ul>
  )
}

/** Tooltip body: the value is the strong element, the name is secondary. */
export function TipBox({
  title,
  rows,
}: {
  title?: React.ReactNode
  rows: { label: string; value: string; color?: string; dashed?: boolean }[]
}) {
  return (
    <div className="min-w-36 rounded-xl bg-popover px-3 py-2.5 text-sm text-popover-foreground shadow-lg ring-1 ring-border">
      {title != null && <p className="mb-1.5 font-medium">{title}</p>}
      <ul className="grid gap-1">
        {rows.map((row) => (
          <li key={row.label} className="flex items-center justify-between gap-5">
            <span className="flex items-center gap-2 text-muted-foreground">
              {row.color && <Swatch color={row.color} dashed={row.dashed} />}
              {row.label}
            </span>
            <span className="font-semibold tabular-nums">{row.value}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

const NOT_DRAWN = Symbol("not drawn")

/**
 * Animation props for one Recharts series, so that it animates only while its
 * data is new. Recharts otherwise re-runs the animation on every resize, which
 * makes a chart lag behind the window or the sidebar while they move.
 *
 * `data` must keep the same identity between renders (memoise it): Recharts
 * also restarts the animation whenever it is handed a new array.
 */
export function useDrawOnce(data: unknown) {
  const [drawn, setDrawn] = React.useState<unknown>(NOT_DRAWN)
  return {
    isAnimationActive: drawn !== data,
    onAnimationEnd: () => setDrawn(data),
  }
}

/** A data point drawn as a glowing dot with a ring in the surface colour. */
export function glowDot(color: string, radius = 4.5) {
  return function GlowDot(props: { cx?: number; cy?: number; index?: number }) {
    if (props.cx == null || props.cy == null) return <g key={props.index} />
    return (
      <circle
        key={props.index}
        className="chart-dot"
        cx={props.cx}
        cy={props.cy}
        r={radius}
        fill={color}
        stroke="var(--card)"
        strokeWidth={2}
      />
    )
  }
}

export const activeDot = (color: string) => ({
  r: 5.5,
  fill: color,
  stroke: "var(--card)",
  strokeWidth: 2,
  className: "chart-dot",
})

/**
 * Sizes a Recharts chart and gives it an accessible summary. With `fill`, the
 * chart is at least `height` tall and grows to use any spare room in its card.
 */
export function ChartFrame({
  label,
  height,
  fill = false,
  className,
  children,
}: {
  label: string
  height: number
  fill?: boolean
  className?: string
  children: React.ReactElement
}) {
  if (fill) {
    return (
      <div
        role="img"
        aria-label={label}
        className={cn("relative w-full flex-1", className)}
        style={{ minHeight: height }}
      >
        <div className="absolute inset-0">
          <ResponsiveContainer width="100%" height="100%">
            {children}
          </ResponsiveContainer>
        </div>
      </div>
    )
  }
  return (
    <div role="img" aria-label={label} className={cn("w-full", className)} style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        {children}
      </ResponsiveContainer>
    </div>
  )
}

/** The same numbers as the chart, as a table for screen readers. */
export function HiddenTable({
  caption,
  columns,
  rows,
}: {
  caption: string
  columns: string[]
  rows: (string | number)[][]
}) {
  return (
    <div className="sr-only">
      <table>
        <caption>{caption}</caption>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column} scope="col">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) =>
                j === 0 ? (
                  <th key={j} scope="row">
                    {cell}
                  </th>
                ) : (
                  <td key={j}>{cell}</td>
                )
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const NICE_STEPS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25]

/**
 * A zoomed-in axis for scores that all sit close together near the top
 * (only used for lines and dots, never for bars, which must start at zero).
 */
export function zoomAxis(low: number, high = 1, maxTicks = 6) {
  const span = Math.max(high - low, 1e-6)
  const step = NICE_STEPS.find((s) => span / s <= maxTicks - 1) ?? 0.25
  const digits = Math.max(0, (String(step).split(".")[1] ?? "").length - 2)
  const round = (value: number) => Number(value.toFixed(6))
  const min = Math.max(0, round(Math.floor(round(low / step)) * step))
  const max = Math.min(1, round(Math.ceil(round(high / step)) * step))
  const ticks: number[] = []
  for (let tick = min; tick <= max + 1e-9; tick = round(tick + step)) ticks.push(tick)
  return { domain: [min, max] as [number, number], ticks, digits }
}
