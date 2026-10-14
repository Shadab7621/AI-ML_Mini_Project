import { Activity } from "lucide-react"
import { motion } from "motion/react"

import { StatusChip } from "@/components/common"
import type { CrossValidation, Problem, TrainResult } from "@/lib/api"
import { formatInt, formatPct, formatScore, statusOf } from "@/lib/format"
import type { Resource } from "@/state/app-state"
import { cn } from "@/lib/utils"

const CARD = "squircle rounded-[28px] bg-card text-card-foreground ring-1 ring-border"
const EYEBROW = "text-[13px] font-semibold tracking-[0.1em] text-muted-foreground uppercase"

/** Five thin bars, one per cross-validation fold, each starting at zero. */
function FoldBars({ cv }: { cv: Resource<CrossValidation> }) {
  const scores = cv.data?.scores
  return (
    <div className="grid gap-2 @sm:justify-items-end">
      <div
        className={cn(
          "flex h-12 items-end gap-1.5 transition-opacity duration-300",
          cv.status !== "ready" && "opacity-45"
        )}
        aria-hidden
      >
        {(scores ?? [0.6, 0.6, 0.6, 0.6, 0.6]).map((score, i) => (
          <motion.span
            key={i}
            className={cn("w-2.5 rounded-full bg-foreground", !scores && "animate-pulse bg-muted-foreground/50")}
            initial={{ height: 0 }}
            animate={{ height: `${Math.max(score, 0.04) * 100}%` }}
            transition={{ type: "spring", stiffness: 220, damping: 24, delay: i * 0.05 }}
          />
        ))}
      </div>
      <p className="text-[13px] leading-snug text-muted-foreground @sm:text-right">
        {cv.data ? (
          <>
            5-fold check: {formatPct(cv.data.mean)}
            <br />± {(cv.data.std * 100).toFixed(2)} pts between folds
          </>
        ) : cv.status === "error" ? (
          "5-fold check failed"
        ) : (
          "Running the 5-fold check…"
        )}
      </p>
    </div>
  )
}

/**
 * The four headline numbers of a trained model. The big card leads with the
 * number that matters most: how often the outcome being detected was missed.
 */
export function ResultsBento({
  result,
  cv,
  problem,
}: {
  result: TrainResult
  cv: Resource<CrossValidation>
  problem: Problem | undefined
}) {
  const [, [missed, caught]] = result.confusion_matrix
  const total = missed + caught
  const [negative, positive] = result.class_names
  const unit = problem?.dataset.unit ?? "rows"
  const { metrics } = result
  const auc = metrics["ROC-AUC"]

  return (
    <div className="grid grid-cols-1 gap-4 @3xl:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
      <div className="surface-invert squircle relative flex min-h-[300px] flex-col justify-between gap-8 overflow-hidden rounded-[32px] bg-(--hero) p-7 text-foreground sm:p-9 @3xl:row-span-2">
        <div
          aria-hidden
          className="absolute -top-px right-0 h-3/5 w-3/4 opacity-25 [background:repeating-linear-gradient(-45deg,transparent_0_13px,var(--foreground)_13px_14px)] [mask-image:radial-gradient(120%_100%_at_100%_0%,black,transparent_68%)]"
        />
        <span className="relative w-fit max-w-full truncate rounded-full bg-foreground/10 px-3 py-1 text-[13px] font-semibold tracking-[0.1em] uppercase">
          Missed: {positive}
        </span>
        <div className="relative grid gap-4">
          <p className="text-[84px] leading-[0.9] font-light tracking-[-0.04em] sm:text-[104px]">
            {formatInt(missed)}
          </p>
          <p className="max-w-[28rem] text-base leading-snug text-muted-foreground">
            of the {formatInt(total)} test {unit} labelled{" "}
            <span className="font-medium text-foreground">{positive}</span> were predicted as{" "}
            <span className="font-medium text-foreground">{negative}</span>.
            {problem?.dataset.miss_note && ` ${problem.dataset.miss_note}`}
          </p>
          <StatusChip status={missed === 0 ? "good" : "bad"}>
            {missed === 0
              ? "Every one of them was caught"
              : `${formatPct(missed / total)} of them slipped through`}
          </StatusChip>
        </div>
      </div>

      <div className={cn(CARD, "@container p-6 sm:p-7")}>
        <div className="grid h-full grid-cols-1 items-center gap-x-6 gap-y-5 @sm:grid-cols-[minmax(0,1fr)_auto]">
          <div className="grid gap-2">
            <p className={EYEBROW}>Accuracy</p>
            <p className="text-[44px] leading-none font-normal tracking-[-0.03em]">
              {formatPct(metrics.Accuracy)}
            </p>
            <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
              <StatusChip status={statusOf(metrics.Accuracy)} />
              on {formatInt(result.y_true.length)} unseen {unit}
            </div>
          </div>
          <FoldBars cv={cv} />
        </div>
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.7fr)] gap-4">
        <div className={cn(CARD, "grid place-content-center justify-items-center gap-2 p-5 text-center")}>
          <p className="text-[32px] leading-none font-normal tracking-[-0.02em]">
            {formatPct(metrics["F1 Score"])}
          </p>
          <p className={EYEBROW}>F1 score</p>
        </div>
        <div className={cn(CARD, "flex items-center gap-4 p-5 sm:p-6")}>
          <span className="grid size-12 shrink-0 place-items-center rounded-full bg-foreground text-background">
            <Activity className="size-6" aria-hidden />
          </span>
          <div className="grid gap-0.5">
            <p className="text-[22px] leading-tight font-semibold tracking-[-0.01em]">
              {auc === null ? "n/a" : formatScore(auc, 3)}
              <span className="font-normal text-muted-foreground"> / 1.000</span>
            </p>
            <p className="text-sm text-muted-foreground">ROC-AUC: how well it ranks risk</p>
          </div>
        </div>
      </div>
    </div>
  )
}
