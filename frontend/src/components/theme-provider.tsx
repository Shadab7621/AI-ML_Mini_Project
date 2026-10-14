import * as React from "react"
import { flushSync } from "react-dom"

export type Theme = "dark" | "light"

const STORAGE_KEY = "theme"

const ThemeContext = React.createContext<
  { theme: Theme; toggleTheme: () => void } | undefined
>(undefined)

function storedTheme(): Theme {
  try {
    return localStorage.getItem(STORAGE_KEY) === "light" ? "light" : "dark"
  } catch {
    return "dark"
  }
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = React.useState<Theme>(storedTheme)

  React.useLayoutEffect(() => {
    const root = document.documentElement
    root.classList.toggle("dark", theme === "dark")
    root.classList.toggle("light", theme === "light")
  }, [theme])

  const toggleTheme = React.useCallback(() => {
    const next: Theme = theme === "dark" ? "light" : "dark"
    try {
      localStorage.setItem(STORAGE_KEY, next)
    } catch {
      // private mode: the choice just won't be remembered
    }
    const apply = () => flushSync(() => setTheme(next))
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches
    // Cross-fade the whole page between themes where the browser can.
    if (document.startViewTransition && !reduced) document.startViewTransition(apply)
    else apply()
  }, [theme])

  const value = React.useMemo(() => ({ theme, toggleTheme }), [theme, toggleTheme])

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export const useTheme = () => {
  const context = React.useContext(ThemeContext)
  if (context === undefined) {
    throw new Error("useTheme must be used within a ThemeProvider")
  }
  return context
}
