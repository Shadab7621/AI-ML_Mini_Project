import * as React from "react"
import { MotionConfig, useAnimate, useReducedMotion } from "motion/react"

import { ErrorNote, Working } from "@/components/common"
import { Backdrop } from "@/components/layout/backdrop"
import { Header } from "@/components/layout/header"
import { Sidebar } from "@/components/layout/sidebar"
import { TooltipProvider } from "@/components/ui/tooltip"
import { ChartsPage } from "@/pages/charts-page"
import { ComparePage } from "@/pages/compare-page"
import { CrossValidationPage } from "@/pages/cv-page"
import { DatasetPage } from "@/pages/dataset-page"
import { ResultsPage } from "@/pages/results-page"
import { ActiveContext } from "@/state/active"
import { AppProvider, TABS, useApp, useTab, type TabId } from "@/state/app-state"

// Pages take no props: memoised, they re-render only when the state they read changes.
const PAGES: Record<TabId, React.ComponentType> = {
  dataset: React.memo(DatasetPage),
  results: React.memo(ResultsPage),
  charts: React.memo(ChartsPage),
  cv: React.memo(CrossValidationPage),
  compare: React.memo(ComparePage),
}

/**
 * Holds one page. A page is built the first time its tab is opened and then kept:
 * switching back only un-hides it, which is what makes tab changes smooth. The
 * page on screen slides and fades in from the side its tab is on. A hidden page
 * keeps its width (so its charts stay laid out) but takes no room and no input.
 */
function PageSlot({
  active,
  direction,
  children,
}: {
  active: boolean
  direction: number
  children: React.ReactNode
}) {
  const [scope, animate] = useAnimate<HTMLDivElement>()
  const reduced = useReducedMotion()

  React.useLayoutEffect(() => {
    if (!active) return
    const controls = animate(
      scope.current,
      { opacity: [0, 1], x: reduced ? 0 : [32 * direction, 0] },
      { duration: 0.28, ease: [0.22, 1, 0.36, 1] }
    )
    return () => controls.stop()
  }, [active, direction, reduced, animate, scope])

  return (
    <div
      ref={scope}
      inert={!active}
      aria-hidden={active ? undefined : true}
      // its own layer on the graphics card, so the slide and fade do not repaint the page
      className={active ? "will-change-[transform,opacity]" : "invisible absolute inset-x-0 top-0 h-0 overflow-hidden"}
    >
      <ActiveContext value={active}>{children}</ActiveContext>
    </div>
  )
}

function Shell() {
  const { meta, dataset, reload } = useApp()
  const { tab, direction } = useTab()
  const [sidebarOpen, setSidebarOpen] = React.useState(
    () => window.matchMedia("(min-width: 1024px)").matches
  )

  // The docked sidebar sticks just below the floating header, whatever its height.
  const header = React.useRef<HTMLElement>(null)
  const [headerHeight, setHeaderHeight] = React.useState(76)
  React.useLayoutEffect(() => {
    const node = header.current
    if (!node) return
    const observer = new ResizeObserver(() => setHeaderHeight(node.offsetHeight))
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  // The Python server needs a few seconds to start; say so if the wait gets noticeable.
  const connecting = meta.status === "loading" && !meta.data
  const [slow, setSlow] = React.useState(false)
  React.useEffect(() => {
    if (!connecting) return
    const timer = window.setTimeout(() => setSlow(true), 2500)
    return () => {
      window.clearTimeout(timer)
      setSlow(false)
    }
  }, [connecting])

  // Green and red only where they mean something: when the outcome being detected is
  // the bad one (or the good one). Otherwise the two outcomes are ink and grey.
  const tone = dataset.data?.dataset.positive_is ?? "neutral"
  const [classNegative, classPositive] =
    tone === "bad"
      ? ["var(--good)", "var(--bad)"]
      : tone === "good"
        ? ["var(--bad)", "var(--good)"]
        : ["var(--chart-3)", "var(--foreground)"]

  // the tabs opened so far, in tab order
  const [visited, setVisited] = React.useState<TabId[]>([tab])
  if (!visited.includes(tab)) setVisited([...visited, tab])

  return (
    <div
      className="relative isolate min-h-svh"
      style={
        {
          "--header-h": `${headerHeight}px`,
          "--class-neg": classNegative,
          "--class-pos": classPositive,
        } as React.CSSProperties
      }
    >
      <Backdrop />
      <Header
        ref={header}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen((open) => !open)}
      />
      <div className="mx-auto flex max-w-[1680px] items-start px-4 pb-20 sm:px-6">
        <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
        <main id="page" role="tabpanel" aria-labelledby={`tab-${tab}`} className="@container relative min-w-0 flex-1 pt-2">
          {connecting && slow && (
            <div className="mb-4">
              <Working>Waiting for the Python server to start…</Working>
            </div>
          )}
          {meta.status === "error" && (
            <div className="mb-5">
              <ErrorNote
                message="Can't reach the Python server. Start it with start.bat. This page keeps trying and connects by itself once the server is running."
                onRetry={reload}
              />
            </div>
          )}
          {TABS.filter((item) => item.id === tab || visited.includes(item.id)).map((item) => {
            const Page = PAGES[item.id]
            return (
              <PageSlot key={item.id} active={item.id === tab} direction={direction}>
                <Page />
              </PageSlot>
            )
          })}
        </main>
      </div>
    </div>
  )
}

export function App() {
  return (
    <MotionConfig reducedMotion="user">
      <TooltipProvider delayDuration={150}>
        <AppProvider>
          <Shell />
        </AppProvider>
      </TooltipProvider>
    </MotionConfig>
  )
}

export default App
