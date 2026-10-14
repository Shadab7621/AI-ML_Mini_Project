import type * as React from "react"
import { Play, Sparkles } from "lucide-react"

import { EmptyState, ErrorNote } from "@/components/common"
import { GlassButton } from "@/components/glass"
import { useActions } from "@/state/actions"
import { useApp } from "@/state/app-state"

/** What the model pages show until a model has been trained. */
export function NeedsModel({
  what,
  skeleton,
}: {
  /** e.g. "its scores" — completes "Train a model to see …". */
  what: string
  skeleton: React.ReactNode
}) {
  const { result, classifier } = useApp()
  const { trainAndShow } = useActions()

  if (result.status === "loading") return skeleton

  return (
    <div className="grid gap-4">
      {result.status === "error" && (
        <ErrorNote message={result.error ?? "Training failed."} onRetry={trainAndShow} />
      )}
      <EmptyState
        icon={Sparkles}
        title="No model trained yet"
        action={
          <GlassButton tone="prominent" size="lg" disabled={!classifier} onClick={trainAndShow}>
            <Play aria-hidden />
            {classifier ? `Train ${classifier.short}` : "Train"}
          </GlassButton>
        }
      >
        Pick a model and its settings on the left, then train it to see {what}.
      </EmptyState>
    </div>
  )
}
