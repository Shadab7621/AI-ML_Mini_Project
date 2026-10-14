export const formatInt = (value: number) => value.toLocaleString("en-US")

/** 0.9594 -> "95.9%" */
export const formatPct = (value: number, digits = 1) =>
  `${(value * 100).toFixed(digits)}%`

/** 0.9594 -> "0.9594" */
export const formatScore = (value: number, digits = 4) => value.toFixed(digits)

export const formatSeconds = (seconds: number) =>
  seconds < 1 ? `${Math.round(seconds * 1000)} ms` : `${seconds.toFixed(1)} s`

export type Status = "good" | "warn" | "bad"

/** Same bands the original app used to colour its metric cards. */
export const statusOf = (value: number, lo = 0.75, hi = 0.88): Status =>
  value >= hi ? "good" : value >= lo ? "warn" : "bad"

export const STATUS_LABEL: Record<Status, string> = {
  good: "Strong",
  warn: "Fair",
  bad: "Weak",
}

/** "heart disease" -> "Heart disease" */
export const capitalise = (text: string) => text.charAt(0).toUpperCase() + text.slice(1)
