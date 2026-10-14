import type * as React from "react"
import { Clock } from "lucide-react"

import { Working } from "@/components/common"
import { Badge } from "@/components/ui/badge"
import type { TrainResult } from "@/lib/api"
import { formatSeconds } from "@/lib/format"

export function PageHeader({
  title,
  description,
  children,
}: {
  title: string
  description?: React.ReactNode
  children?: React.ReactNode
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-3 pt-1 pb-1">
      <div className="grid gap-1.5">
        <h1 className="text-[28px] leading-tight font-semibold tracking-[-0.02em]">{title}</h1>
        {description && (
          <p className="max-w-3xl text-base text-muted-foreground">{description}</p>
        )}
      </div>
      {children}
    </div>
  )
}

const formatParam = (value: number | string) =>
  typeof value === "number" && !Number.isInteger(value) ? String(Number(value.toFixed(4))) : String(value)

/** Which model the numbers on the page belong to, and with which settings. */
export function ModelSummary({ result, updating }: { result: TrainResult; updating: boolean }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2 text-sm text-muted-foreground">
      <Badge className="h-7 px-3 text-sm">{result.classifier}</Badge>
      {Object.entries(result.params).map(([key, value]) => (
        <span key={key} className="rounded-full bg-muted px-2.5 py-1 text-[13px]">
          {key.replaceAll("_", " ")}{" "}
          <span className="font-medium text-foreground">{formatParam(value)}</span>
        </span>
      ))}
      {updating ? (
        <Working>Training the new model…</Working>
      ) : (
        <span className="inline-flex items-center gap-1.5">
          <Clock className="size-4" aria-hidden />
          trained in {formatSeconds(result.train_seconds)}
        </span>
      )}
    </div>
  )
}
