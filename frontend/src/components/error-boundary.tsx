import * as React from "react"

import { ErrorNote } from "@/components/common"

/**
 * Catches a crash inside one page so that the tab bar and the sidebar keep
 * working, and offers a way back instead of a blank screen.
 */
export class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error) {
    console.error("A page crashed:", error)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="pt-4">
          <ErrorNote
            message={`This page ran into a problem (${this.state.error.message}). Your data and models are safe.`}
            onRetry={() => this.setState({ error: null })}
          />
        </div>
      )
    }
    return this.props.children
  }
}
