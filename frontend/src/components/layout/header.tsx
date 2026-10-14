import * as React from "react"
import {
  BrainCircuit,
  ChartSpline,
  Database,
  Moon,
  PanelLeft,
  Repeat2,
  Sun,
  Target,
  Trophy,
  type LucideIcon,
} from "lucide-react"
import { AnimatePresence, motion } from "motion/react"

import { GlassButton, GlassPanel } from "@/components/glass"
import { useTheme } from "@/components/theme-provider"
import { capitalise } from "@/lib/format"
import { TABS, useApp, useTab, type TabId } from "@/state/app-state"
import { cn } from "@/lib/utils"

const TAB_ICONS: Record<TabId, LucideIcon> = {
  dataset: Database,
  results: Target,
  charts: ChartSpline,
  cv: Repeat2,
  compare: Trophy,
}

function TabBar() {
  const { setTab } = useApp()
  const { tab } = useTab()
  const buttons = React.useRef<(HTMLButtonElement | null)[]>([])

  const onKeyDown = (event: React.KeyboardEvent) => {
    const step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0
    if (!step) return
    event.preventDefault()
    const index = TABS.findIndex((item) => item.id === tab)
    const next = (index + step + TABS.length) % TABS.length
    setTab(TABS[next].id)
    buttons.current[next]?.focus()
  }

  return (
    <GlassPanel refract tint={0.3} className="h-[52px] max-w-full rounded-full">
      <div
        role="tablist"
        aria-label="Sections"
        onKeyDown={onKeyDown}
        className="flex items-center gap-1 overflow-x-auto px-1.5 [scrollbar-width:none]"
      >
        {TABS.map((item, index) => {
          const active = item.id === tab
          const Icon = TAB_ICONS[item.id]
          return (
            <button
              key={item.id}
              ref={(node) => {
                buttons.current[index] = node
              }}
              type="button"
              role="tab"
              id={`tab-${item.id}`}
              aria-selected={active}
              aria-controls="page"
              tabIndex={active ? 0 : -1}
              onClick={() => setTab(item.id)}
              className={cn(
                "relative isolate inline-flex h-10 shrink-0 items-center gap-2 rounded-full px-4 text-[15px] font-medium outline-none transition-colors duration-200 focus-visible:ring-3 focus-visible:ring-ring",
                active ? "text-background" : "text-muted-foreground hover:text-foreground"
              )}
            >
              {active && (
                <motion.span
                  layoutId="tab-pill"
                  className="absolute inset-0 -z-10 rounded-full bg-foreground shadow-[inset_0_1px_0_0_rgb(255_255_255/0.45),0_4px_14px_-4px_rgb(0_0_0/0.45)]"
                  transition={{ type: "spring", stiffness: 380, damping: 32 }}
                />
              )}
              <Icon className="size-[18px]" aria-hidden />
              <span className={cn(!active && "max-sm:sr-only")}>{item.label}</span>
            </button>
          )
        })}
      </div>
    </GlassPanel>
  )
}

function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()
  const Icon = theme === "dark" ? Sun : Moon
  return (
    <GlassButton
      size="icon"
      className="size-[52px]"
      onClick={toggleTheme}
      aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.span
          key={theme}
          className="grid place-items-center"
          initial={{ rotate: -70, scale: 0.4, opacity: 0 }}
          animate={{ rotate: 0, scale: 1, opacity: 1 }}
          exit={{ rotate: 70, scale: 0.4, opacity: 0 }}
          transition={{ duration: 0.18 }}
        >
          <Icon aria-hidden />
        </motion.span>
      </AnimatePresence>
    </GlassButton>
  )
}

export function Header({
  ref,
  sidebarOpen,
  onToggleSidebar,
}: {
  ref?: React.Ref<HTMLElement>
  sidebarOpen: boolean
  onToggleSidebar: () => void
}) {
  const { dataset } = useApp()
  const problem = dataset.data
  const subtitle = problem
    ? (problem.dataset.tagline ?? `${capitalise(problem.dataset.name)}: predicting ${problem.target}`)
    : "Train and compare classifiers"

  return (
    <header ref={ref} className="pointer-events-none sticky top-0 z-30">
      {/* content fades out as it scrolls up behind the floating glass */}
      <div
        aria-hidden
        className="absolute inset-x-0 top-0 -z-10 h-[calc(100%+1rem)] bg-linear-to-b from-background from-25% via-background/75 to-transparent"
      />
      <motion.div
        layoutRoot
        className="mx-auto grid max-w-[1680px] grid-cols-[minmax(0,1fr)_auto] items-center gap-3 px-4 py-3 sm:px-6 xl:grid-cols-[auto_minmax(0,1fr)_auto] min-[1440px]:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)]"
      >
        <div className="flex min-w-0 items-center gap-2 *:pointer-events-auto">
          <GlassButton
            size="icon"
            className="size-[52px]"
            onClick={onToggleSidebar}
            aria-label={sidebarOpen ? "Hide model settings" : "Show model settings"}
            aria-expanded={sidebarOpen}
          >
            <PanelLeft aria-hidden />
          </GlassButton>
          <GlassPanel tint={0.3} className="h-[52px] min-w-0 rounded-full">
            <div className="flex min-w-0 items-center gap-3 pr-5 pl-2">
              <span className="grid size-9 shrink-0 place-items-center rounded-full bg-foreground text-background">
                <BrainCircuit className="size-5" aria-hidden />
              </span>
              <div className="min-w-0 leading-tight">
                <p className="truncate text-[15px] font-semibold tracking-[-0.01em]">
                  Classification Visualiser
                </p>
                <p className="truncate text-[13px] text-muted-foreground max-sm:hidden">
                  {subtitle}
                </p>
              </div>
            </div>
          </GlassPanel>
        </div>

        <nav className="pointer-events-auto order-last col-span-2 min-w-0 justify-self-center xl:order-none xl:col-span-1">
          <TabBar />
        </nav>

        <div className="pointer-events-auto justify-self-end">
          <ThemeToggle />
        </div>
      </motion.div>
    </header>
  )
}
