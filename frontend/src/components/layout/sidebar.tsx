import * as React from "react"
import { ChevronRight, Info, Play, RotateCcw, Trophy, X } from "lucide-react"
import { AnimatePresence, motion } from "motion/react"

import { Segmented } from "@/components/common"
import { GlassButton } from "@/components/glass"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Slider } from "@/components/ui/slider"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { ParamSpec } from "@/lib/api"
import { capitalise, formatInt } from "@/lib/format"
import { useActions } from "@/state/actions"
import { useApp } from "@/state/app-state"
import { cn } from "@/lib/utils"

type NumberSpec = Extract<ParamSpec, { type: "int" | "float" }>

const decimalsOf = (value: number) => (String(value).split(".")[1] ?? "").length

/** Every value the slider can stop on: the minimum, then round multiples of the step, then the maximum. */
function stopsOf(spec: NumberSpec) {
  const digits = Math.max(decimalsOf(spec.step), decimalsOf(spec.min))
  const round = (value: number) => Number(value.toFixed(digits))
  const stops = [round(spec.min)]
  for (let k = Math.floor(spec.min / spec.step) + 1; round(k * spec.step) < spec.max; k++) {
    const stop = round(k * spec.step)
    if (stop > spec.min) stops.push(stop)
  }
  stops.push(round(spec.max))
  return stops
}

function Help({ text }: { text: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label="What does this do?"
          className="grid size-5 place-items-center rounded-full text-muted-foreground outline-none hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring"
        >
          <Info className="size-4" aria-hidden />
        </button>
      </TooltipTrigger>
      <TooltipContent side="right" sideOffset={6} className="max-w-64 text-sm leading-snug">
        {text}
      </TooltipContent>
    </Tooltip>
  )
}

function NumberControl({
  spec,
  value,
  note,
  onChange,
}: {
  spec: NumberSpec
  value: number
  /** Shown under the slider when the setting has no effect on the current data. */
  note?: string
  onChange: (value: number) => void
}) {
  const stops = React.useMemo(() => stopsOf(spec), [spec])
  const index = stops.reduce(
    (best, stop, i) => (Math.abs(stop - value) < Math.abs(stops[best] - value) ? i : best),
    0
  )
  const shown =
    spec.type === "int" ? formatInt(value) : value.toFixed(decimalsOf(spec.step))
  const id = React.useId()

  return (
    <div className={cn("grid gap-2.5", note && "opacity-60")}>
      <div className="flex items-center gap-1.5">
        <label id={id} className="text-[15px] font-medium">
          {spec.label}
        </label>
        {spec.help && <Help text={spec.help} />}
        <output className="ml-auto rounded-md bg-muted px-2 py-0.5 text-sm font-medium tabular-nums">
          {shown}
        </output>
      </div>
      <Slider
        aria-labelledby={id}
        min={0}
        max={stops.length - 1}
        step={1}
        value={[index]}
        onValueChange={([next]) => onChange(stops[next])}
      />
      <div className="flex justify-between text-[13px] text-muted-foreground tabular-nums">
        <span>{stops[0]}</span>
        <span>{formatInt(stops[stops.length - 1])}</span>
      </div>
      {note && <p className="text-[13px] leading-snug text-muted-foreground">{note}</p>}
    </div>
  )
}

/** Which data the models work on; a click goes to the page where it is chosen. */
function DataSummary({ onNavigate }: { onNavigate?: () => void }) {
  const { dataset, setTab } = useApp()
  const problem = dataset.data
  if (!problem) return null
  return (
    <button
      type="button"
      onClick={() => {
        setTab("dataset")
        onNavigate?.()
      }}
      className="group flex w-full items-center gap-3 rounded-2xl bg-muted/60 px-3.5 py-3 text-left outline-none transition-colors hover:bg-muted focus-visible:ring-3 focus-visible:ring-ring"
    >
      <span className="grid min-w-0 flex-1 gap-0.5">
        <span className="text-[13px] font-semibold tracking-[0.08em] text-muted-foreground uppercase">
          Data
        </span>
        <span className="truncate text-[15px] font-medium">{capitalise(problem.dataset.name)}</span>
        <span className="truncate text-[13px] text-muted-foreground">
          Predicting {problem.target}: {problem.class_names[1]} or {problem.class_names[0]}
        </span>
      </span>
      <ChevronRight
        className="size-5 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5"
        aria-hidden
      />
    </button>
  )
}

function Controls({ onNavigate }: { onNavigate?: () => void }) {
  const { meta, dataset, classifier, selectClassifier, params, setParam, resetParams, isDefault } =
    useApp()
  const problem = dataset.data
  const unused = (kind: "number" | "category" | undefined) =>
    problem && kind === "number" && problem.n_numeric === 0
      ? "This data has no number columns, so this setting has no effect."
      : problem && kind === "category" && problem.n_categorical === 0
        ? "This data has no category columns, so this setting has no effect."
        : undefined

  if (meta.status === "error") {
    return (
      <p className="text-[15px] text-muted-foreground">
        The model settings appear here once the Python server is running.
      </p>
    )
  }

  if (!classifier || !meta.data) {
    return (
      <div className="grid gap-6">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="grid gap-3">
            <Skeleton className="h-4 w-28" />
            <Skeleton className="h-10 w-full rounded-xl" />
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="grid gap-7">
      <DataSummary onNavigate={onNavigate} />
      <div className="grid gap-2.5">
        <label id="model-label" className="text-[15px] font-medium">
          Model
        </label>
        <Select value={classifier.name} onValueChange={selectClassifier}>
          <SelectTrigger
            aria-labelledby="model-label"
            className="h-11! w-full rounded-xl px-3.5 text-[15px] font-medium"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent position="popper" className="rounded-xl">
            {meta.data.classifiers.map((spec) => (
              <SelectItem key={spec.name} value={spec.name} className="py-2 text-[15px]">
                {spec.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-[13px] text-muted-foreground">
          {classifier.scaled
            ? "Number columns are standardised before training."
            : "Number columns are used on their own scale."}
        </p>
      </div>

      <div className="grid gap-5">
        <div className="flex h-7 items-center justify-between">
          <h3 className="text-[13px] font-semibold tracking-[0.08em] text-muted-foreground uppercase">
            Settings
          </h3>
          {!isDefault && (
            <Button variant="ghost" size="sm" className="h-7 text-sm" onClick={resetParams}>
              <RotateCcw aria-hidden />
              Reset
            </Button>
          )}
        </div>
        {classifier.params.map((spec) =>
          spec.type === "choice" ? (
            <div key={spec.key} className="grid gap-2.5">
              <div className="flex items-center gap-1.5">
                <span className="text-[15px] font-medium">{spec.label}</span>
                {spec.help && <Help text={spec.help} />}
              </div>
              <Segmented
                label={spec.label}
                value={String(params[spec.key] ?? spec.default)}
                options={spec.options.map((option) => ({ value: option, label: option }))}
                onChange={(next) => setParam(spec.key, next)}
              />
            </div>
          ) : (
            <NumberControl
              key={spec.key}
              spec={spec}
              value={Number(params[spec.key] ?? spec.default)}
              note={unused(spec.applies_to)}
              onChange={(next) => setParam(spec.key, next)}
            />
          )
        )}
      </div>
    </div>
  )
}

function SidebarPanel({ onClose }: { onClose?: () => void }) {
  const { dataset, result, comparison, dirty, canTrain } = useApp()
  const problem = dataset.data
  const { trainAndShow, compareAndShow } = useActions()
  const training = result.status === "loading"
  const comparing = comparison.status === "loading"

  const onTrain = () => {
    trainAndShow()
    onClose?.()
  }
  const onCompare = () => {
    compareAndShow()
    onClose?.()
  }

  return (
    <aside
      aria-label="Model settings"
      className="squircle flex h-full flex-col overflow-hidden rounded-[24px] bg-card ring-1 ring-border"
    >
      <motion.div layoutScroll className="min-h-0 flex-1 overflow-y-auto p-5 [scrollbar-width:thin]">
        {onClose && (
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-[17px] font-semibold">Model settings</h2>
            <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close">
              <X aria-hidden />
            </Button>
          </div>
        )}
        <Controls onNavigate={onClose} />
      </motion.div>

      <div className="grid gap-2.5 border-t p-4">
        {dirty && !training && (
          <p className="text-[13px] text-muted-foreground">
            Settings changed. Train again to update the results.
          </p>
        )}
        <GlassButton
          tone="prominent"
          size="lg"
          className="w-full"
          disabled={!canTrain}
          loading={training}
          onClick={onTrain}
        >
          {!training && <Play aria-hidden />}
          {training ? "Training…" : "Train & evaluate"}
        </GlassButton>
        <GlassButton className="w-full" disabled={!canTrain} loading={comparing} onClick={onCompare}>
          {!comparing && <Trophy aria-hidden />}
          {comparing ? "Comparing…" : "Compare all 7 models"}
        </GlassButton>
        <p className="pt-1 text-center text-[13px] leading-snug text-muted-foreground">
          {problem && (
            <>
              {formatInt(problem.n_samples)} {problem.dataset.unit} · {problem.n_features} features
              <br />
            </>
          )}
          70 / 30 train-test split · seed 42
        </p>
      </div>
    </aside>
  )
}

const DESKTOP = "(min-width: 1024px)"

export function useIsDesktop() {
  const [desktop, setDesktop] = React.useState(() => window.matchMedia(DESKTOP).matches)
  React.useEffect(() => {
    const query = window.matchMedia(DESKTOP)
    const onChange = () => setDesktop(query.matches)
    query.addEventListener("change", onChange)
    return () => query.removeEventListener("change", onChange)
  }, [])
  return desktop
}

/** Docked beside the page on wide screens; a slide-over drawer on small ones. */
export function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  const desktop = useIsDesktop()

  if (desktop) {
    return (
      <motion.div
        layoutRoot
        initial={false}
        animate={{ width: open ? 344 : 0, opacity: open ? 1 : 0 }}
        transition={{ type: "spring", stiffness: 300, damping: 34 }}
        inert={!open}
        className="sticky top-[calc(var(--header-h)+8px)] h-[calc(100svh-var(--header-h)-24px)] shrink-0 overflow-hidden"
      >
        <div className="h-full w-[344px] py-px pr-6 pl-px">
          <SidebarPanel />
        </div>
      </motion.div>
    )
  }

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            key="scrim"
            className="fixed inset-0 z-40 bg-black/55"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.div
            key="drawer"
            layoutRoot
            className="fixed inset-y-3 left-3 z-50 w-[min(344px,calc(100vw-24px))]"
            initial={{ x: -380 }}
            animate={{ x: 0 }}
            exit={{ x: -380 }}
            transition={{ type: "spring", stiffness: 320, damping: 34 }}
          >
            <SidebarPanel onClose={onClose} />
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
