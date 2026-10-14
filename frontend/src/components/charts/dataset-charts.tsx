// Charts for the Dataset page: class balance, feature-to-target links,
// per-feature distributions and the association heatmap.
import * as React from "react"
import { motion } from "motion/react"
import { Bar, BarChart, CartesianGrid, Pie, PieChart, Tooltip, XAxis, YAxis } from "recharts"

import {
  AXIS_TICK,
  ChartFrame,
  GRID,
  HiddenTable,
  INK,
  Legend,
  TipBox,
  useDrawOnce,
  type TipProps,
} from "@/components/charts/kit"
import type { Problem } from "@/lib/api"
import { featureLabel, valueLabel } from "@/lib/features"
import { capitalise, formatInt, formatPct, formatScore } from "@/lib/format"

/** The two outcomes, in the order [everything else, the outcome being detected]. */
export const CLASS_COLORS = ["var(--class-neg)", "var(--class-pos)"]

const compact = (value: number) =>
  value >= 1000 ? `${(value / 1000).toFixed(value % 1000 === 0 ? 0 : 1)}K` : String(value)

/* ── Class balance ────────────────────────────────────────────────────────── */

export function ClassDonut({ counts, unit }: { counts: Problem["class_counts"]; unit: string }) {
  const total = counts.reduce((sum, item) => sum + item.count, 0)
  const slices = React.useMemo(
    () => counts.map((item, i) => ({ ...item, fill: CLASS_COLORS[i] ?? INK })),
    [counts]
  )
  const description = `Class balance: ${counts
    .map((item) => `${formatInt(item.count)} ${item.label}`)
    .join(", ")}.`
  const draw = useDrawOnce(slices)

  return (
    <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-5">
      <div className="relative size-[188px] shrink-0">
        <ChartFrame label={description} height={188}>
          <PieChart>
            <Pie
              data={slices}
              dataKey="count"
              nameKey="label"
              innerRadius="66%"
              outerRadius="100%"
              paddingAngle={3}
              cornerRadius={7}
              stroke="none"
              startAngle={90}
              endAngle={-270}
              {...draw}
              animationDuration={800}
            />
            <Tooltip
              content={(tip: TipProps) => {
                const slice = tip.active ? tip.payload?.[0]?.payload : null
                if (!slice) return null
                return (
                  <TipBox
                    rows={[
                      {
                        label: slice.label,
                        value: `${formatInt(slice.count)} (${formatPct(slice.count / total)})`,
                      },
                    ]}
                  />
                )
              }}
            />
          </PieChart>
        </ChartFrame>
        <div className="pointer-events-none absolute inset-0 grid place-content-center text-center">
          <span className="text-[13px] text-muted-foreground">{capitalise(unit)}</span>
          <span className="text-2xl font-semibold tracking-[-0.02em]">{formatInt(total)}</span>
        </div>
      </div>
      <ul className="grid min-w-52 flex-1 gap-3">
        {slices.map((slice) => (
          <li key={slice.label} className="flex items-center justify-between gap-4">
            <span className="flex min-w-0 items-center gap-2.5 text-[15px]">
              <span className="size-3 shrink-0 rounded-[4px]" style={{ background: slice.fill }} />
              <span className="truncate">{capitalise(slice.label)}</span>
            </span>
            <span className="text-[15px] whitespace-nowrap tabular-nums">
              <span className="font-semibold">{formatInt(slice.count)}</span>
              <span className="ml-2 text-muted-foreground">{formatPct(slice.count / total)}</span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/* ── Which features are most related to the target ────────────────────────── */

export function TargetLinkList({ features }: { features: Problem["top_features"] }) {
  return (
    <ol className="flex flex-1 flex-col justify-between gap-5">
      {features.map((item, i) => (
        <li key={item.feature} className="grid gap-2.5">
          <div className="flex items-baseline justify-between gap-3">
            <span className="truncate text-[15px] font-medium">{featureLabel(item.feature)}</span>
            <span className="text-[17px] font-semibold tabular-nums">
              {formatScore(item.strength, 2)}
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-muted" aria-hidden>
            <motion.div
              className="h-full rounded-full bg-foreground"
              initial={{ width: 0 }}
              animate={{ width: `${item.strength * 100}%` }}
              transition={{ duration: 0.7, delay: 0.15 + i * 0.05, ease: [0.22, 1, 0.36, 1] }}
            />
          </div>
        </li>
      ))}
    </ol>
  )
}

/* ── How each top feature splits between the two outcomes ─────────────────── */

function DistributionChart({
  item,
  classNames,
  dataset,
}: {
  item: Problem["top_features"][number]
  classNames: [string, string]
  dataset: string
}) {
  const rows = React.useMemo(
    () =>
      item.values.map((entry) => ({
        ...entry,
        name: item.kind === "number" ? entry.value : valueLabel(dataset, item.feature, entry.value),
      })),
    [item, dataset]
  )
  const longest = Math.max(...rows.map((row) => row.name.length))
  const crowded = rows.length > 5 || longest * rows.length > 48
  const title = featureLabel(item.feature)
  const description = `How ${title} values split between ${classNames[0]} and ${classNames[1]}.`
  const drawNegative = useDrawOnce(rows)
  const drawPositive = useDrawOnce(rows)

  return (
    <figure className="m-0 grid gap-2">
      <figcaption className="flex items-baseline justify-between gap-3">
        <span className="truncate text-[15px] font-medium">{title}</span>
        <span className="shrink-0 text-[13px] text-muted-foreground">
          link to target {formatScore(item.strength, 2)}
        </span>
      </figcaption>
      <ChartFrame label={description} height={crowded ? 250 : 206}>
        <BarChart
          data={rows}
          margin={{ top: 8, right: 4, bottom: 0, left: 0 }}
          barGap={2}
          barCategoryGap="24%"
        >
          <CartesianGrid vertical={false} stroke={GRID} />
          <XAxis
            dataKey="name"
            interval={0}
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={{ stroke: GRID }}
            tickMargin={crowded ? 4 : 8}
            angle={crowded ? -40 : 0}
            textAnchor={crowded ? "end" : "middle"}
            height={crowded ? Math.min(96, 22 + longest * 4.6) : 30}
          />
          <YAxis
            tickFormatter={compact}
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={false}
            width={38}
          />
          <Tooltip
            cursor={{ fill: "var(--muted)", opacity: 0.55 }}
            content={(tip: TipProps) => {
              const row = tip.active ? tip.payload?.[0]?.payload : null
              if (!row) return null
              return (
                <TipBox
                  title={`${title}: ${row.name}`}
                  rows={[
                    { label: capitalise(classNames[0]), value: formatInt(row.negative), color: CLASS_COLORS[0] },
                    { label: capitalise(classNames[1]), value: formatInt(row.positive), color: CLASS_COLORS[1] },
                  ]}
                />
              )
            }}
          />
          <Bar
            dataKey="negative"
            fill={CLASS_COLORS[0]}
            radius={[4, 4, 0, 0]}
            maxBarSize={22}
            {...drawNegative}
            animationDuration={600}
          />
          <Bar
            dataKey="positive"
            fill={CLASS_COLORS[1]}
            radius={[4, 4, 0, 0]}
            maxBarSize={22}
            {...drawPositive}
            animationDuration={600}
          />
        </BarChart>
      </ChartFrame>
      <HiddenTable
        caption={description}
        columns={[title, classNames[0], classNames[1]]}
        rows={rows.map((row) => [row.name, row.negative, row.positive])}
      />
    </figure>
  )
}

export function DistributionGrid({
  features,
  classNames,
  dataset,
}: {
  features: Problem["top_features"]
  classNames: [string, string]
  dataset: string
}) {
  return (
    <div className="flex flex-col gap-5">
      <Legend
        items={[
          { label: capitalise(classNames[0]), color: CLASS_COLORS[0], kind: "box" },
          { label: capitalise(classNames[1]), color: CLASS_COLORS[1], kind: "box" },
        ]}
      />
      <div className="grid grid-cols-1 gap-x-8 gap-y-8 @2xl:grid-cols-2 @6xl:grid-cols-3">
        {features.map((item) => (
          <DistributionChart key={item.feature} item={item} classNames={classNames} dataset={dataset} />
        ))}
      </div>
    </div>
  )
}

/* ── Association heatmap ──────────────────────────────────────────────────── */

const GAP = 2

/**
 * Lower triangle of the table of how strongly each pair of columns is related
 * (0 = not at all, 1 = one fully determines the other). Darker (lighter in the
 * dark theme) means stronger. The first row and column are the target.
 */
export function AssociationHeatmap({ association }: { association: Problem["association"] }) {
  const { labels, matrix, hidden } = association
  const names = labels.map((label, i) => (i === 0 ? `${featureLabel(label)} (target)` : featureLabel(label)))
  const size = labels.length - 1 // rows 1..n-1 against columns 0..n-2
  const [hover, setHover] = React.useState<{ row: number; col: number } | null>(null)

  // fewer columns get bigger squares; the label gutters fit the longest name
  const cell = size <= 12 ? 26 : size <= 22 ? 20 : 16
  const pitch = cell + GAP
  const longest = Math.max(...names.map((name) => name.length))
  const left = Math.min(230, Math.round(longest * 6.4) + 18)
  const bottom = Math.min(170, Math.round(longest * 5) + 22)
  const width = left + size * pitch
  const height = size * pitch + bottom
  const value = hover ? matrix[hover.row]?.[hover.col] ?? null : null
  const roomy = size >= 14 // enough empty space above the diagonal for the scale

  const scale = (
    <div className="grid gap-1.5 text-[13px] text-muted-foreground">
      <span>Strength of the link</span>
      <span className="flex items-center gap-2">
        none
        <span
          className="h-2.5 w-28 rounded-full ring-1 ring-border"
          style={{ background: "linear-gradient(to right, var(--muted), var(--foreground))" }}
        />
        perfect
      </span>
    </div>
  )

  return (
    <div className="flex flex-col gap-4">
      <div className="overflow-x-auto pb-1 [scrollbar-width:thin]">
        <div className="relative w-fit" onPointerLeave={() => setHover(null)}>
          <svg
            width={width}
            height={height}
            role="img"
            aria-label="Heatmap of how strongly each pair of columns is related."
            className="block"
          >
            {Array.from({ length: size }, (_, r) => {
              const row = r + 1
              return (
                <g key={row}>
                  <text
                    x={left - 10}
                    y={r * pitch + cell / 2}
                    textAnchor="end"
                    dominantBaseline="central"
                    fontSize={12}
                    fill={hover?.row === row ? "var(--foreground)" : "var(--muted-foreground)"}
                    fontWeight={hover?.row === row ? 600 : 400}
                  >
                    {names[row]}
                  </text>
                  {Array.from({ length: row }, (_, col) => {
                    const strength = matrix[row][col]
                    const active = hover?.row === row && hover.col === col
                    return (
                      <g key={col} onPointerEnter={() => setHover({ row, col })}>
                        <rect
                          x={left + col * pitch}
                          y={r * pitch}
                          width={cell}
                          height={cell}
                          rx={4}
                          fill="var(--muted)"
                        />
                        {strength !== null && (
                          <rect
                            x={left + col * pitch}
                            y={r * pitch}
                            width={cell}
                            height={cell}
                            rx={4}
                            fill={INK}
                            fillOpacity={Math.min(1, Math.abs(strength))}
                          />
                        )}
                        {active && (
                          <rect
                            x={left + col * pitch - 1.5}
                            y={r * pitch - 1.5}
                            width={cell + 3}
                            height={cell + 3}
                            rx={5.5}
                            fill="none"
                            stroke="var(--foreground)"
                            strokeWidth={2}
                          />
                        )}
                        {/* a hit area that covers the gap, so moving between cells never drops the tooltip */}
                        <rect
                          x={left + col * pitch - GAP / 2}
                          y={r * pitch - GAP / 2}
                          width={pitch}
                          height={pitch}
                          fill="transparent"
                        />
                      </g>
                    )
                  })}
                </g>
              )
            })}
            {Array.from({ length: size }, (_, col) => (
              <text
                key={col}
                transform={`translate(${left + col * pitch + cell / 2 + 3}, ${size * pitch + 8}) rotate(-50)`}
                textAnchor="end"
                dominantBaseline="central"
                fontSize={12}
                fill={hover?.col === col ? "var(--foreground)" : "var(--muted-foreground)"}
                fontWeight={hover?.col === col ? 600 : 400}
              >
                {names[col]}
              </text>
            ))}
          </svg>

          {hover && hover.row <= size && (
            <div
              className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full pb-2"
              style={{
                left: left + hover.col * pitch + cell / 2,
                top: (hover.row - 1) * pitch,
              }}
            >
              <TipBox
                title={`${names[hover.row]} × ${names[hover.col]}`}
                rows={[
                  {
                    label: "Strength",
                    value: value === null ? "n/a" : formatScore(value, 2),
                  },
                ]}
              />
            </div>
          )}

          {roomy && <div className="absolute top-0 right-0 text-right [&_span]:justify-end">{scale}</div>}
        </div>
      </div>
      {!roomy && scale}
      {hidden > 0 && (
        <p className="text-[13px] text-muted-foreground">
          Showing the {size} columns most related to the target. {hidden} more are not shown.
        </p>
      )}
    </div>
  )
}
