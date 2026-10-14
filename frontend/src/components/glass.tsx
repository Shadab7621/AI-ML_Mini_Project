import * as React from "react"
import { LoaderCircle } from "lucide-react"
import { motion, type HTMLMotionProps } from "motion/react"

import GlassSurface from "@/components/GlassSurface"
import { cn } from "@/lib/utils"

/**
 * The liquid-glass material as a layer that fills its parent. The parent must be
 * `relative isolate`; content goes above it with `z-10`.
 *
 * With `refract` it is React Bits' GlassSurface, which bends whatever is behind it.
 * That is expensive (the browser redoes it whenever the backdrop changes), so it is
 * kept for the one place where content scrolls underneath. Everywhere else the
 * glass is drawn with plain CSS.
 */
export function GlassLayer({
  radius = 999,
  tint = 0.08,
  refract = false,
}: {
  radius?: number
  tint?: number
  refract?: boolean
}) {
  if (!refract) {
    return <span aria-hidden className="glass-lite absolute inset-0 rounded-[inherit]" />
  }
  return (
    <GlassSurface
      width="100%"
      height="100%"
      borderRadius={radius}
      backgroundOpacity={tint}
      saturation={1}
      distortionScale={-130}
      displace={0.6}
      redOffset={0}
      greenOffset={0}
      blueOffset={0}
      style={{ position: "absolute", inset: 0 }}
    />
  )
}

export function GlassPanel({
  className,
  radius,
  tint,
  refract,
  children,
  ...props
}: React.ComponentProps<"div"> & { radius?: number; tint?: number; refract?: boolean }) {
  return (
    <div className={cn("relative isolate", className)} {...props}>
      <GlassLayer radius={radius} tint={tint} refract={refract} />
      <div className="relative z-10 flex h-full items-center">{children}</div>
    </div>
  )
}

type GlassButtonProps = Omit<HTMLMotionProps<"button">, "children"> & {
  /** "prominent" is the filled call-to-action; "clear" is see-through glass. */
  tone?: "clear" | "prominent"
  size?: "md" | "lg" | "icon"
  loading?: boolean
  children?: React.ReactNode
}

const SIZES = {
  md: "h-11 px-5 text-[15px]",
  lg: "h-12 px-6 text-base",
  icon: "size-11",
}

export function GlassButton({
  tone = "clear",
  size = "md",
  loading = false,
  disabled,
  className,
  children,
  onPointerMove,
  ...props
}: GlassButtonProps) {
  return (
    <motion.button
      type="button"
      disabled={disabled || loading}
      whileHover={{ scale: 1.015 }}
      whileTap={{ scale: 0.955 }}
      transition={{ type: "spring", stiffness: 420, damping: 24 }}
      onPointerMove={(event) => {
        // the highlight under the pointer (see .glass-sheen)
        const box = event.currentTarget.getBoundingClientRect()
        event.currentTarget.style.setProperty("--px", `${event.clientX - box.left}px`)
        event.currentTarget.style.setProperty("--py", `${event.clientY - box.top}px`)
        onPointerMove?.(event)
      }}
      className={cn(
        "group/glass relative isolate inline-flex shrink-0 items-center justify-center rounded-full font-medium whitespace-nowrap outline-none select-none focus-visible:ring-3 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-55",
        tone === "prominent" ? "text-background" : "text-foreground",
        SIZES[size],
        className
      )}
      {...props}
    >
      <GlassLayer />
      {tone === "prominent" && (
        <span
          aria-hidden
          className="absolute inset-0 rounded-full bg-foreground/90 shadow-[inset_0_1px_0_0_rgb(255_255_255/0.4),inset_0_-1px_0_0_rgb(0_0_0/0.2)]"
        />
      )}
      <span
        aria-hidden
        className="glass-sheen pointer-events-none absolute inset-0 rounded-full opacity-0 transition-opacity duration-300 group-hover/glass:opacity-100"
      />
      <span className="relative z-10 inline-flex items-center gap-2 [&_svg]:size-[18px] [&_svg]:shrink-0">
        {loading && <LoaderCircle className="animate-spin" aria-hidden />}
        {children}
      </span>
    </motion.button>
  )
}
