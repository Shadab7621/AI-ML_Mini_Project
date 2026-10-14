import * as React from "react"
import {
  CircleAlert,
  CircleCheck,
  CircleX,
  LoaderCircle,
  TriangleAlert,
  type LucideIcon,
} from "lucide-react"
import { motion } from "motion/react"

import { GlassButton } from "@/components/glass"
import { STATUS_LABEL, type Status } from "@/lib/format"
import type { Resource } from "@/state/app-state"
import { cn } from "@/lib/utils"

/** The card every section sits in. */
export function Panel({
  title,
  subtitle,
  actions,
  className,
  children,
  ...props
}: Omit<React.ComponentProps<"section">, "title"> & {
  title?: React.ReactNode
  subtitle?: React.ReactNode
  actions?: React.ReactNode
}) {
  return (
    <section
      className={cn(
        "squircle flex min-w-0 flex-col gap-5 rounded-[24px] bg-card p-5 text-card-foreground ring-1 ring-border sm:p-6",
        className
      )}
      {...props}
    >
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
          <div className="grid gap-1">
            <h2 className="text-[17px] leading-snug font-semibold tracking-[-0.01em]">{title}</h2>
            {subtitle && <p className="text-sm text-muted-foreground">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  )
}

const STATUS_STYLE: Record<Status, { className: string; icon: LucideIcon }> = {
  good: { className: "bg-good/12 text-good-ink", icon: CircleCheck },
  warn: { className: "bg-warn/18 text-warn-ink", icon: TriangleAlert },
  bad: { className: "bg-bad/12 text-bad-ink", icon: CircleX },
}

/** Status is never colour alone: it always carries an icon and a word. */
export function StatusChip({
  status,
  children,
  className,
}: {
  status: Status
  children?: React.ReactNode
  className?: string
}) {
  const { className: tone, icon: Icon } = STATUS_STYLE[status]
  return (
    <span
      className={cn(
        "inline-flex min-h-6 w-fit items-center gap-1 rounded-full px-2 py-0.5 text-left text-[13px] leading-tight font-medium",
        tone,
        className
      )}
    >
      <Icon className="size-3.5 shrink-0" aria-hidden />
      {children ?? STATUS_LABEL[status]}
    </span>
  )
}

export function StatTile({
  label,
  value,
  hint,
  icon: Icon,
  status,
  className,
}: {
  label: string
  value: React.ReactNode
  hint?: React.ReactNode
  icon?: LucideIcon
  status?: Status
  className?: string
}) {
  return (
    <div
      className={cn(
        "squircle grid content-start gap-2 rounded-[22px] bg-card p-5 text-card-foreground ring-1 ring-border",
        className
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-medium text-muted-foreground">{label}</p>
        {Icon && <Icon className="size-[18px] text-muted-foreground" aria-hidden />}
      </div>
      <p className="text-[30px] leading-none font-semibold tracking-[-0.02em]">{value}</p>
      {(hint || status) && (
        <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
          {status && <StatusChip status={status} />}
          {hint}
        </div>
      )}
    </div>
  )
}

/** Holds the previous render at reduced opacity while a new answer is on its way. */
export function Stale({
  when,
  className,
  children,
}: {
  when: boolean
  className?: string
  children: React.ReactNode
}) {
  return (
    <div
      aria-busy={when}
      className={cn("transition-opacity duration-300", when && "opacity-45", className)}
    >
      {children}
    </div>
  )
}

export function Working({ children }: { children: React.ReactNode }) {
  return (
    <p role="status" className="inline-flex items-center gap-2 text-sm text-muted-foreground">
      <LoaderCircle className="size-4 animate-spin" aria-hidden />
      {children}
    </p>
  )
}

export function EmptyState({
  icon: Icon,
  title,
  children,
  action,
}: {
  icon: LucideIcon
  title: string
  children?: React.ReactNode
  action?: React.ReactNode
}) {
  return (
    <div className="squircle grid justify-items-center gap-4 rounded-[24px] bg-card px-6 py-16 text-center ring-1 ring-border">
      <span className="grid size-14 place-items-center rounded-full bg-muted">
        <Icon className="size-6" aria-hidden />
      </span>
      <div className="grid max-w-md gap-1.5">
        <h2 className="text-xl font-semibold tracking-[-0.01em]">{title}</h2>
        {children && <p className="text-base text-muted-foreground">{children}</p>}
      </div>
      {action}
    </div>
  )
}

export function ErrorNote({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div
      role="alert"
      className="flex flex-wrap items-center gap-x-4 gap-y-3 rounded-2xl bg-bad/10 px-4 py-3 text-[15px] ring-1 ring-bad/30"
    >
      <CircleAlert className="size-5 shrink-0 text-bad-ink" aria-hidden />
      <p className="min-w-0 flex-1">{message}</p>
      {onRetry && (
        <GlassButton className="h-9 px-4 text-sm" onClick={onRetry}>
          Try again
        </GlassButton>
      )}
    </div>
  )
}

/** A row of options where exactly one is picked; the highlight slides between them. */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
  className,
}: {
  value: T
  options: readonly { value: T; label: string }[]
  onChange: (value: T) => void
  label: string
  className?: string
}) {
  const id = React.useId()
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cn("grid auto-cols-fr grid-flow-col rounded-full bg-muted p-1", className)}
    >
      {options.map((option) => {
        const active = option.value === value
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(option.value)}
            className={cn(
              "relative isolate h-8 min-w-0 rounded-full px-1.5 text-sm font-medium outline-none transition-colors focus-visible:ring-3 focus-visible:ring-ring",
              active ? "text-foreground" : "text-muted-foreground hover:text-foreground"
            )}
          >
            {active && (
              <motion.span
                layoutId={id}
                className="absolute inset-0 -z-10 rounded-full bg-card shadow-sm ring-1 ring-border"
                transition={{ type: "spring", stiffness: 480, damping: 36 }}
              />
            )}
            <span className="block truncate">{option.label}</span>
          </button>
        )
      })}
    </div>
  )
}

/** Sections of a page fade up one after another. */
export function Reveal({
  index = 0,
  className,
  children,
}: {
  index?: number
  className?: string
  children: React.ReactNode
}) {
  return (
    <motion.div
      className={cn("min-w-0", className)}
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.06 + index * 0.06, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  )
}

/**
 * A slot for something that takes a while to compute. Shows a waiting note first,
 * then the content; when a newer answer is on its way the old one stays, dimmed.
 */
export function Pending<T>({
  resource,
  height,
  waiting,
  onRetry,
  children,
}: {
  resource: Resource<T>
  /** Height of the waiting box, matched to the content so nothing jumps. */
  height: number
  waiting: string
  onRetry: () => void
  children: (data: T) => React.ReactNode
}) {
  if (resource.status === "error" && !resource.data) {
    return <ErrorNote message={resource.error ?? "Something went wrong."} onRetry={onRetry} />
  }
  if (!resource.data) {
    return (
      <div
        className="grid place-items-center rounded-2xl bg-muted/40 px-6 text-center"
        style={{ minHeight: height }}
      >
        <Working>{waiting}</Working>
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-3">
      {resource.status === "error" && (
        <ErrorNote message={resource.error ?? "Something went wrong."} onRetry={onRetry} />
      )}
      <Stale when={resource.status !== "ready"}>{children(resource.data)}</Stale>
    </div>
  )
}
