import * as React from "react"

/**
 * Pages stay alive after the first visit so that coming back to a tab is instant.
 * This tells a page whether it is the one on screen: a hidden page must not start
 * slow analyses nobody is looking at.
 */
export const ActiveContext = React.createContext(true)

/**
 * Runs `load` while the page it sits in is on screen. It is a component of its
 * own (it draws nothing) so that showing or hiding a page re-renders only this,
 * not the page and its charts.
 */
export function WhenActive({ load }: { load: () => void }) {
  const active = React.useContext(ActiveContext)
  React.useEffect(() => {
    if (active) load()
  }, [active, load])
  return null
}
