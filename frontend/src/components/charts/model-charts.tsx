// The Recharts charts for a trained model: ROC / precision-recall curves, learning
// curve, feature importance, cross-validation folds and the model leaderboard.
import * as React from "react"
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  ErrorBar,
  Label,
  LabelList,
  Line,
  ReferenceArea,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import {
  AXIS_LABEL,
  AXIS_TICK,
  ChartFrame,
  GRID,
  HiddenTable,
  INK,
  INK_2,
  Legend,
  TipBox,
  activeDot,
  glowDot,
  useDrawOnce,
  zoomAxis,
  type TipProps,
} from "@/components/charts/kit"
import type { CrossValidation, FeatureImportance, LearningCurve } from "@/lib/api"
import { featureLabel } from "@/lib/features"
import { formatInt, formatPct, formatScore } from "@/lib/format"

const UNIT_TICKS = [0, 0.25, 0.5, 0.75, 1]

/* ── ROC and precision-recall ─────────────────────────────────────────────── */

export function CurveChart({
  x,
  y,
  xLabel,
  yLabel,
  step = false,
  diagonal = false,
  description,
}: {
  x: number[]
  y: number[]
  xLabel: string
  yLabel: string
  /** Draw as steps (precision-recall) instead of straight segments (ROC). */
  step?: boolean
  /** The "random guess" line of a ROC plot. */
  diagonal?: boolean
  description: string
}) {
  const id = React.useId().replace(/:/g, "")
  const points = React.useMemo(() => x.map((value, i) => ({ x: value, y: y[i] })), [x, y])
  const curve = step ? "stepAfter" : "linear"
  const draw = useDrawOnce(points)

  return (
    <>
      <ChartFrame label={description} height={300} fill>
        <ComposedChart data={points} margin={{ top: 10, right: 14, bottom: 24, left: 6 }}>
          <defs>
            <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={INK} stopOpacity={0.16} />
              <stop offset="100%" stopColor={INK} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} stroke={GRID} />
          <XAxis
            dataKey="x"
            type="number"
            domain={[0, 1]}
            ticks={UNIT_TICKS}
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={{ stroke: GRID }}
            tickMargin={8}
          >
            <Label value={xLabel} position="insideBottom" offset={-16} {...AXIS_LABEL} />
          </XAxis>
          <YAxis
            type="number"
            domain={[0, 1]}
            ticks={UNIT_TICKS}
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={false}
            width={52}
          >
            <Label
              value={yLabel}
              angle={-90}
              position="insideLeft"
              offset={8}
              style={{ textAnchor: "middle" }}
              {...AXIS_LABEL}
            />
          </YAxis>
          <Tooltip
            cursor={{ stroke: INK_2, strokeWidth: 1 }}
            content={(tip: TipProps) => {
              const point = tip.active ? tip.payload?.[0]?.payload : null
              if (!point) return null
              return (
                <TipBox
                  rows={[
                    { label: yLabel, value: formatScore(point.y, 3), color: INK },
                    { label: xLabel, value: formatScore(point.x, 3) },
                  ]}
                />
              )
            }}
          />
          {diagonal && (
            <ReferenceLine
              segment={[
                { x: 0, y: 0 },
                { x: 1, y: 1 },
              ]}
              stroke={INK_2}
              strokeWidth={1.5}
              strokeDasharray="5 5"
            />
          )}
          <Area
            type={curve}
            dataKey="y"
            stroke="none"
            fill={`url(#${id})`}
            activeDot={false}
            tooltipType="none"
            isAnimationActive={false}
          />
          <Line
            className="chart-glow"
            type={curve}
            dataKey="y"
            stroke={INK}
            strokeWidth={2.25}
            strokeLinecap="round"
            strokeLinejoin="round"
            dot={points.length <= 12 ? glowDot(INK) : false}
            activeDot={activeDot(INK)}
            {...draw}
            animationDuration={900}
          />
        </ComposedChart>
      </ChartFrame>
      <HiddenTable
        caption={description}
        columns={[xLabel, yLabel]}
        rows={points.map((point) => [formatScore(point.x, 3), formatScore(point.y, 3)])}
      />
    </>
  )
}

/* ── Learning curve ───────────────────────────────────────────────────────── */

/** Mean ± 1 standard deviation, kept inside 0–1. */
const band = (mean: number, std: number) => [Math.max(0, mean - std), Math.min(1, mean + std)]

export function LearningCurveChart({ data }: { data: LearningCurve }) {
  const rows = React.useMemo(
    () =>
      data.sizes.map((size, i) => ({
        size,
        train: data.train_mean[i],
        cv: data.cv_mean[i],
        trainBand: band(data.train_mean[i], data.train_std[i]),
        cvBand: band(data.cv_mean[i], data.cv_std[i]),
      })),
    [data]
  )
  // zoom the axis to where the scores are, but never tighter than 2 points
  const highest = Math.max(...rows.map((row) => Math.max(row.trainBand[1], row.cvBand[1])))
  const lowest = Math.min(...rows.map((row) => Math.min(row.trainBand[0], row.cvBand[0])))
  const axis = zoomAxis(Math.min(lowest, highest - 0.02), highest)
  const description =
    "Training accuracy and cross-validation accuracy as the training set grows."
  const drawTraining = useDrawOnce(rows)
  const drawValidation = useDrawOnce(rows)

  return (
    <div className="grid gap-3">
      <Legend
        items={[
          { label: "Training accuracy", color: INK },
          { label: "Cross-validation accuracy", color: INK_2, dashed: true },
        ]}
      />
      <ChartFrame label={description} height={320}>
        <ComposedChart data={rows} margin={{ top: 10, right: 16, bottom: 24, left: 6 }}>
          <CartesianGrid vertical={false} stroke={GRID} />
          <XAxis
            dataKey="size"
            type="number"
            domain={["dataMin", "dataMax"]}
            ticks={data.sizes}
            tickFormatter={formatInt}
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={{ stroke: GRID }}
            tickMargin={8}
            padding={{ left: 12, right: 12 }}
          >
            <Label value="Training set size" position="insideBottom" offset={-16} {...AXIS_LABEL} />
          </XAxis>
          <YAxis
            type="number"
            domain={axis.domain}
            ticks={axis.ticks}
            tickFormatter={(value: number) => formatPct(value, axis.digits)}
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={false}
            width={58}
          />
          <Tooltip
            cursor={{ stroke: INK_2, strokeWidth: 1 }}
            content={(tip: TipProps) => {
              const row = tip.active ? tip.payload?.[0]?.payload : null
              if (!row) return null
              return (
                <TipBox
                  title={`${formatInt(row.size)} training samples`}
                  rows={[
                    { label: "Training", value: formatPct(row.train, 2), color: INK },
                    { label: "Cross-validation", value: formatPct(row.cv, 2), color: INK_2, dashed: true },
                  ]}
                />
              )
            }}
          />
          <Area
            type="monotone"
            dataKey="trainBand"
            stroke="none"
            fill={INK}
            fillOpacity={0.1}
            activeDot={false}
            tooltipType="none"
            isAnimationActive={false}
          />
          <Area
            type="monotone"
            dataKey="cvBand"
            stroke="none"
            fill={INK_2}
            fillOpacity={0.14}
            activeDot={false}
            tooltipType="none"
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="cv"
            stroke={INK_2}
            strokeWidth={2.25}
            strokeDasharray="6 5"
            strokeLinecap="round"
            dot={glowDot(INK_2, 4)}
            activeDot={activeDot(INK_2)}
            {...drawValidation}
            animationDuration={900}
          />
          <Line
            className="chart-glow"
            type="monotone"
            dataKey="train"
            stroke={INK}
            strokeWidth={2.25}
            strokeLinecap="round"
            dot={glowDot(INK)}
            activeDot={activeDot(INK)}
            {...drawTraining}
            animationDuration={900}
          />
        </ComposedChart>
      </ChartFrame>
      <HiddenTable
        caption={description}
        columns={["Training set size", "Training accuracy", "Cross-validation accuracy"]}
        rows={rows.map((row) => [row.size, formatPct(row.train, 2), formatPct(row.cv, 2)])}
      />
    </div>
  )
}

/* ── Feature importance ───────────────────────────────────────────────────── */

export function ImportanceChart({ data }: { data: FeatureImportance }) {
  const permutation = data.method === "permutation"
  const rows = React.useMemo(
    () =>
      data.items.map((item) => ({
        ...item,
        label: featureLabel(item.feature),
        std: item.std ?? 0,
      })),
    [data]
  )
  const description = permutation
    ? "Drop in accuracy when each feature is shuffled; larger means more important."
    : "Share of the model's splits each feature is responsible for."
  const draw = useDrawOnce(rows)

  // The value sits just past the bar tip (and past the error whisker when there is one).
  const valueLabel = (props: {
    x?: number | string
    y?: number | string
    width?: number | string
    height?: number | string
    index?: number
  }) => {
    const row = rows[props.index ?? -1]
    if (!row) return null
    const x = Number(props.x)
    const width = Number(props.width)
    const whisker = row.importance > 0 ? (row.std / row.importance) * Math.max(width, 0) : 0
    return (
      <text
        x={x + Math.max(width, 0) + whisker + 8}
        y={Number(props.y) + Number(props.height) / 2}
        dominantBaseline="central"
        fill="var(--foreground)"
        fontSize={13}
        className="tabular-nums"
      >
        {formatScore(row.importance, 3)}
      </text>
    )
  }

  return (
    <>
      <ChartFrame label={description} height={rows.length * 30 + 36}>
        <BarChart
          data={rows}
          layout="vertical"
          margin={{ top: 2, right: 64, bottom: 0, left: 0 }}
          barCategoryGap={7}
        >
          <CartesianGrid horizontal={false} stroke={GRID} />
          <XAxis
            type="number"
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value: number) => formatScore(value, 2)}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={182}
            interval={0}
            tick={{ fill: "var(--foreground)", fontSize: 13 }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            cursor={{ fill: "var(--muted)", opacity: 0.55 }}
            content={(tip: TipProps) => {
              const row = tip.active ? tip.payload?.[0]?.payload : null
              if (!row) return null
              return (
                <TipBox
                  title={row.label}
                  rows={[
                    {
                      label: permutation ? "Accuracy drop" : "Importance",
                      value: permutation
                        ? `${formatScore(row.importance, 4)} ± ${formatScore(row.std, 4)}`
                        : formatScore(row.importance, 4),
                    },
                  ]}
                />
              )
            }}
          />
          <Bar
            dataKey="importance"
            fill={INK}
            radius={[0, 4, 4, 0]}
            maxBarSize={16}
            {...draw}
            animationDuration={700}
          >
            {permutation && (
              <ErrorBar dataKey="std" direction="x" width={5} strokeWidth={1.5} stroke={INK_2} />
            )}
            <LabelList content={valueLabel} />
          </Bar>
        </BarChart>
      </ChartFrame>
      <HiddenTable
        caption={description}
        columns={["Feature", permutation ? "Accuracy drop" : "Importance"]}
        rows={rows.map((row) => [row.label, formatScore(row.importance, 4)])}
      />
    </>
  )
}

/* ── Cross-validation folds ───────────────────────────────────────────────── */

export function FoldChart({ data }: { data: CrossValidation }) {
  const rows = React.useMemo(
    () =>
      data.scores.map((score, i) => ({
        fold: `Fold ${i + 1}`,
        accuracy: score,
        stem: [data.mean, score],
      })),
    [data]
  )
  const low = Math.min(data.min, data.mean - data.std)
  const high = Math.max(data.max, data.mean + data.std)
  const pad = Math.max((high - low) * 0.3, 0.002)
  const axis = zoomAxis(low - pad, Math.min(1, high + pad))
  const description = "Accuracy of each cross-validation fold, compared with the average."

  return (
    <div className="grid gap-3">
      <Legend
        items={[
          { label: "Fold accuracy", color: INK },
          { label: `Average ${formatPct(data.mean, 2)}`, color: INK_2, dashed: true },
          { label: "Average ± 1 standard deviation", color: "color-mix(in oklab, var(--foreground) 18%, transparent)", kind: "box" },
        ]}
      />
      <ChartFrame label={description} height={320}>
        <ComposedChart data={rows} margin={{ top: 30, right: 16, bottom: 8, left: 6 }}>
          <CartesianGrid vertical={false} stroke={GRID} />
          <XAxis
            dataKey="fold"
            tick={{ fill: "var(--foreground)", fontSize: 13 }}
            tickLine={false}
            axisLine={{ stroke: GRID }}
            tickMargin={8}
            padding={{ left: 24, right: 24 }}
          />
          <YAxis
            type="number"
            domain={axis.domain}
            ticks={axis.ticks}
            tickFormatter={(value: number) => formatPct(value, axis.digits)}
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={false}
            width={64}
          />
          <Tooltip
            cursor={false}
            content={(tip: TipProps) => {
              const row = tip.active ? tip.payload?.[0]?.payload : null
              if (!row) return null
              const delta = row.accuracy - data.mean
              return (
                <TipBox
                  title={row.fold}
                  rows={[
                    { label: "Accuracy", value: formatPct(row.accuracy, 2), color: INK },
                    {
                      label: "From average",
                      value: `${delta >= 0 ? "+" : "−"}${Math.abs(delta * 100).toFixed(2)} pts`,
                    },
                  ]}
                />
              )
            }}
          />
          <ReferenceArea
            y1={Math.max(axis.domain[0], data.mean - data.std)}
            y2={Math.min(axis.domain[1], data.mean + data.std)}
            fill={INK}
            fillOpacity={0.08}
            stroke="none"
          />
          <ReferenceLine y={data.mean} stroke={INK_2} strokeWidth={1.5} strokeDasharray="5 5" />
          <Bar dataKey="stem" barSize={2} fill={INK_2} tooltipType="none" isAnimationActive={false} />
          <Line
            dataKey="accuracy"
            stroke="none"
            dot={glowDot(INK, 6)}
            activeDot={{ ...activeDot(INK), r: 7 }}
            isAnimationActive={false}
          >
            <LabelList
              dataKey="accuracy"
              content={(props: { x?: number | string; y?: number | string; value?: unknown }) => {
                const score = Number(props.value)
                const above = score >= data.mean
                return (
                  <text
                    x={Number(props.x)}
                    y={Number(props.y) + (above ? -17 : 19)}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill="var(--foreground)"
                    fontSize={13}
                  >
                    {formatPct(score, 2)}
                  </text>
                )
              }}
            />
          </Line>
        </ComposedChart>
      </ChartFrame>
      <HiddenTable
        caption={description}
        columns={["Fold", "Accuracy"]}
        rows={rows.map((row) => [row.fold, formatPct(row.accuracy, 2)])}
      />
    </div>
  )
}

/* ── Leaderboard: one metric across all models ────────────────────────────── */

export function LeaderboardChart({
  rows,
  metric,
}: {
  /** Keep the same array between renders (memoise it), or the bars re-animate. */
  rows: { model: string; value: number }[]
  metric: string
}) {
  const sorted = React.useMemo(() => [...rows].sort((a, b) => b.value - a.value), [rows])
  const description = `${metric} of every model, best first.`
  const draw = useDrawOnce(sorted)

  return (
    <>
      <ChartFrame label={description} height={sorted.length * 42 + 34}>
        <BarChart
          data={sorted}
          layout="vertical"
          margin={{ top: 2, right: 64, bottom: 0, left: 0 }}
          barCategoryGap={12}
        >
          <CartesianGrid horizontal={false} stroke={GRID} />
          <XAxis
            type="number"
            domain={[0, 1]}
            ticks={UNIT_TICKS}
            tickFormatter={(value: number) => formatPct(value, 0)}
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            type="category"
            dataKey="model"
            width={140}
            interval={0}
            tick={{ fill: "var(--foreground)", fontSize: 14 }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            cursor={{ fill: "var(--muted)", opacity: 0.55 }}
            content={(tip: TipProps) => {
              const row = tip.active ? tip.payload?.[0]?.payload : null
              if (!row) return null
              return (
                <TipBox title={row.model} rows={[{ label: metric, value: formatScore(row.value) }]} />
              )
            }}
          />
          <Bar
            dataKey="value"
            fill={INK}
            radius={[0, 4, 4, 0]}
            maxBarSize={18}
            {...draw}
            animationDuration={600}
          >
            <LabelList
              dataKey="value"
              position="right"
              offset={8}
              fill="var(--foreground)"
              fontSize={13}
              formatter={(value: unknown) => formatScore(Number(value))}
            />
          </Bar>
        </BarChart>
      </ChartFrame>
      <HiddenTable
        caption={description}
        columns={["Model", metric]}
        rows={sorted.map((row) => [row.model, formatScore(row.value)])}
      />
    </>
  )
}
