import * as React from "react"

import {
  api,
  ApiError,
  type ClassifierSpec,
  type Comparison,
  type CrossValidation,
  type DatasetList,
  type FeatureImportance,
  type LearningCurve,
  type Meta,
  type ModelRequest,
  type ParamValues,
  type Problem,
  type ProblemSpec,
  type TrainResult,
} from "@/lib/api"

export const TABS = [
  { id: "dataset", label: "Dataset" },
  { id: "results", label: "Results" },
  { id: "charts", label: "Charts" },
  { id: "cv", label: "Cross-validation" },
  { id: "compare", label: "Compare" },
] as const

export type TabId = (typeof TABS)[number]["id"]

/**
 * Something fetched from the API. `data` can be present while `status` is not
 * "ready": that is the previous answer, kept on screen (dimmed) until the new
 * one arrives so the layout never jumps.
 */
export interface Resource<T> {
  status: "idle" | "loading" | "ready" | "error"
  data?: T
  error?: string
}

type SetResource<T> = React.Dispatch<React.SetStateAction<Resource<T>>>

const IDLE = { status: "idle" } as const

const messageOf = (error: unknown) =>
  error instanceof Error ? error.message : "Something went wrong."

function load<T>(set: SetResource<T>, fetcher: () => Promise<T>, isCurrent: () => boolean) {
  set((prev) => ({ status: "loading", data: prev.data }))
  fetcher().then(
    (data) => {
      if (isCurrent()) set({ status: "ready", data })
    },
    (error: unknown) => {
      if (isCurrent()) set((prev) => ({ status: "error", error: messageOf(error), data: prev.data }))
    }
  )
}

/**
 * Loads something the whole app needs, and keeps trying until it arrives. The
 * Python server takes a few seconds to start and the page may open first, so a
 * failure to connect is reported only after about ten seconds, and even then the
 * attempts go on: once the server is up the page fills in by itself.
 *
 * If the server answers and says no (an ApiError), trying again cannot help:
 * `onRefused` is called once and the loop ends.
 */
async function loadForever<T>(
  set: SetResource<T>,
  fetcher: () => Promise<T>,
  isCurrent: () => boolean,
  onRefused?: (message: string) => void
) {
  set((prev) => ({ status: "loading", data: prev.data }))
  for (let attempt = 1; isCurrent(); attempt++) {
    try {
      const data = await fetcher()
      if (isCurrent()) set({ status: "ready", data })
      return
    } catch (error) {
      if (!isCurrent()) return
      if (error instanceof ApiError) {
        set((prev) => ({ status: "error", error: error.message, data: prev.data }))
        onRefused?.(error.message)
        return
      }
      if (attempt === 7) {
        set((prev) => ({ status: "error", error: messageOf(error), data: prev.data }))
      }
      await new Promise((resolve) => window.setTimeout(resolve, attempt < 7 ? 1500 : 3000))
    }
  }
}

const defaultsOf = (spec: ClassifierSpec): ParamValues =>
  Object.fromEntries(spec.params.map((param) => [param.key, param.default]))

const sameParams = (a: ParamValues, b: ParamValues) => {
  const keys = Object.keys(a)
  return keys.length === Object.keys(b).length && keys.every((key) => a[key] === b[key])
}

const tabFromHash = (): TabId => {
  const id = window.location.hash.slice(1)
  return TABS.some((tab) => tab.id === id) ? (id as TabId) : "dataset"
}

// The data the app opens with, and the choice it falls back to.
const DEFAULT_SPEC: ProblemSpec = { dataset: "mushrooms", target: null, positive: null }
const SPEC_KEY = "problem"

function storedSpec(): ProblemSpec {
  try {
    const saved = JSON.parse(localStorage.getItem(SPEC_KEY) ?? "null")
    if (saved && typeof saved.dataset === "string") {
      return {
        dataset: saved.dataset,
        target: typeof saved.target === "string" ? saved.target : null,
        positive: typeof saved.positive === "string" ? saved.positive : null,
      }
    }
  } catch {
    // nothing saved, or storage is not available
  }
  return DEFAULT_SPEC
}

/**
 * Which tab is open. Kept apart from the rest of the state so that changing tab
 * re-renders only the header and the page frame, not every page and chart.
 */
interface TabState {
  tab: TabId
  /** +1 when the newly opened tab is to the right of the previous one, -1 otherwise. */
  direction: 1 | -1
}

interface AppState {
  setTab: (tab: TabId) => void
  /** The open tab, for code that acts on it without needing to re-render when it changes. */
  getTab: () => TabId

  meta: Resource<Meta>
  /** The datasets that can be chosen: the samples and the uploaded files. */
  datasets: Resource<DatasetList>
  /** The chosen data, target column and outcome, with everything the Dataset page shows. */
  dataset: Resource<Problem>
  /** Set when a choice of data, column or outcome was refused; says why. */
  notice: string | null
  dismissNotice: () => void
  reload: () => void

  chooseDataset: (id: string) => void
  chooseTarget: (column: string) => void
  choosePositive: (value: string) => void
  uploading: boolean
  uploadDataset: (file: File) => void
  removeDataset: (id: string) => void

  classifier: ClassifierSpec | null
  selectClassifier: (name: string) => void
  params: ParamValues
  setParam: (key: string, value: number | string) => void
  resetParams: () => void
  isDefault: boolean

  /** The model the results on screen belong to (or are being computed for). */
  trained: ModelRequest | null
  /** True when the sidebar no longer matches the trained model. */
  dirty: boolean
  result: Resource<TrainResult>
  importance: Resource<FeatureImportance>
  learning: Resource<LearningCurve>
  cv: Resource<CrossValidation>
  comparison: Resource<Comparison>

  /** False until the data is loaded: there is nothing to train on before that. */
  canTrain: boolean
  train: () => void
  compare: () => void
  /** These start the slower analyses for the trained model. Pass true to retry after an error. */
  loadImportance: (retry?: boolean) => void
  loadLearning: (retry?: boolean) => void
  loadCv: (retry?: boolean) => void
}

const AppContext = React.createContext<AppState | undefined>(undefined)
const TabContext = React.createContext<TabState | undefined>(undefined)

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [tab, setTabState] = React.useState<TabId>(tabFromHash)
  const [direction, setDirection] = React.useState<1 | -1>(1)

  const [meta, setMeta] = React.useState<Resource<Meta>>(IDLE)
  const [datasets, setDatasets] = React.useState<Resource<DatasetList>>(IDLE)
  const [dataset, setDataset] = React.useState<Resource<Problem>>(IDLE)
  const [spec, setSpec] = React.useState<ProblemSpec>(storedSpec)
  const [notice, setNotice] = React.useState<string | null>(null)
  const [uploading, setUploading] = React.useState(false)

  const [classifierName, setClassifierName] = React.useState<string | null>(null)
  const [paramsByClassifier, setParamsByClassifier] = React.useState<
    Record<string, ParamValues>
  >({})

  const [trained, setTrained] = React.useState<ModelRequest | null>(null)
  const [result, setResult] = React.useState<Resource<TrainResult>>(IDLE)
  const [importance, setImportance] = React.useState<Resource<FeatureImportance>>(IDLE)
  const [learning, setLearning] = React.useState<Resource<LearningCurve>>(IDLE)
  const [cv, setCv] = React.useState<Resource<CrossValidation>>(IDLE)
  const [comparison, setComparison] = React.useState<Resource<Comparison>>(IDLE)

  // Each training run gets a number; answers that belong to an older run are dropped.
  const run = React.useRef(0)

  const currentTab = React.useRef(tab)
  const setTab = React.useCallback((next: TabId) => {
    const current = currentTab.current
    if (current === next) return
    const index = (id: TabId) => TABS.findIndex((item) => item.id === id)
    setDirection(index(next) > index(current) ? 1 : -1)
    currentTab.current = next
    setTabState(next)
  }, [])
  const getTab = React.useCallback(() => currentTab.current, [])
  const dismissNotice = React.useCallback(() => setNotice(null), [])

  React.useEffect(() => {
    if (window.location.hash.slice(1) !== tab) {
      window.history.replaceState(null, "", `#${tab}`)
    }
    window.scrollTo({ top: 0 })
  }, [tab])

  React.useEffect(() => {
    const onHashChange = () => setTab(tabFromHash())
    window.addEventListener("hashchange", onHashChange)
    return () => window.removeEventListener("hashchange", onHashChange)
  }, [setTab])

  // ── the models and the list of datasets ────────────────────────────────────

  // Each (re)load gets a number, so an older loop stops when a newer one starts.
  const baseLoad = React.useRef(0)
  const reload = React.useCallback(() => {
    const id = ++baseLoad.current
    const isCurrent = () => baseLoad.current === id
    void loadForever(setMeta, api.meta, isCurrent)
    void loadForever(setDatasets, api.datasets, isCurrent)
    return () => {
      if (isCurrent()) baseLoad.current++
    }
  }, [])

  React.useEffect(() => reload(), [reload])

  // ── the chosen data, target column and outcome ─────────────────────────────

  /** Results belong to one problem: when the problem changes they are thrown away. */
  const forgetModels = React.useCallback(() => {
    run.current++
    setTrained(null)
    setResult(IDLE)
    setImportance(IDLE)
    setLearning(IDLE)
    setCv(IDLE)
    setComparison(IDLE)
  }, [])

  // The last choice the server accepted, to go back to when a new one is refused.
  const accepted = React.useRef<ProblemSpec | null>(null)

  React.useEffect(() => {
    let current = true // false once a newer choice has replaced this one
    void loadForever(
      setDataset,
      () => api.problem(spec),
      () => current,
      (message) => {
        // e.g. an uploaded file that was removed, or an outcome with too few rows
        const fallback = accepted.current ?? DEFAULT_SPEC
        const same = JSON.stringify(fallback) === JSON.stringify(spec)
        setNotice(message)
        if (!same) setSpec(fallback)
      }
    )
    try {
      localStorage.setItem(SPEC_KEY, JSON.stringify(spec))
    } catch {
      // private mode: the choice just won't be remembered
    }
    return () => {
      current = false
    }
  }, [spec])

  const choose = React.useCallback(
    (next: ProblemSpec) => {
      setNotice(null)
      forgetModels()
      setSpec(next)
    },
    [forgetModels]
  )

  const chooseDataset = React.useCallback(
    (id: string) => choose({ dataset: id, target: null, positive: null }),
    [choose]
  )
  const chooseTarget = React.useCallback(
    (column: string) => choose({ dataset: spec.dataset, target: column, positive: null }),
    [choose, spec.dataset]
  )
  const shownTarget = dataset.data?.target ?? spec.target
  const choosePositive = React.useCallback(
    (value: string) => choose({ dataset: spec.dataset, target: shownTarget, positive: value }),
    [choose, spec.dataset, shownTarget]
  )

  const uploadDataset = React.useCallback(
    (file: File) => {
      setUploading(true)
      setNotice(null)
      api.upload(file).then(
        ({ id }) => {
          setUploading(false)
          load(setDatasets, api.datasets, () => true)
          chooseDataset(id)
        },
        (error: unknown) => {
          setUploading(false)
          setNotice(messageOf(error))
        }
      )
    },
    [chooseDataset]
  )

  const removeDataset = React.useCallback(
    (id: string) => {
      api.removeDataset(id).then(
        () => {
          load(setDatasets, api.datasets, () => true)
          accepted.current = null
          if (spec.dataset === id) chooseDataset(DEFAULT_SPEC.dataset)
        },
        (error: unknown) => setNotice(messageOf(error))
      )
    },
    [chooseDataset, spec.dataset]
  )

  // ── the model and its settings ─────────────────────────────────────────────

  // Until a model is picked, the first one is selected; until a setting is
  // changed, a model uses its defaults.
  const classifiers = meta.data?.classifiers
  const classifier = React.useMemo(
    () => classifiers?.find((item) => item.name === classifierName) ?? classifiers?.[0] ?? null,
    [classifiers, classifierName]
  )
  const params = React.useMemo<ParamValues>(
    () => (classifier ? (paramsByClassifier[classifier.name] ?? defaultsOf(classifier)) : {}),
    [classifier, paramsByClassifier]
  )
  const isDefault = classifier ? sameParams(params, defaultsOf(classifier)) : true

  const setParam = React.useCallback(
    (key: string, value: number | string) => {
      if (!classifier) return
      setParamsByClassifier((current) => ({
        ...current,
        [classifier.name]: {
          ...(current[classifier.name] ?? defaultsOf(classifier)),
          [key]: value,
        },
      }))
    },
    [classifier]
  )

  const resetParams = React.useCallback(() => {
    if (!classifier) return
    setParamsByClassifier((current) => ({
      ...current,
      [classifier.name]: defaultsOf(classifier),
    }))
  }, [classifier])

  // ── training ───────────────────────────────────────────────────────────────

  // Requests name the problem exactly as the server resolved it.
  const problem = React.useMemo<ProblemSpec | null>(
    () =>
      dataset.status === "ready" && dataset.data
        ? {
            dataset: dataset.data.dataset.id,
            target: dataset.data.target,
            positive: dataset.data.positive,
          }
        : null,
    [dataset]
  )
  React.useEffect(() => {
    if (problem) accepted.current = problem
  }, [problem])
  const canTrain = problem !== null && classifier !== null

  const train = React.useCallback(() => {
    if (!classifier || !problem) return
    const request: ModelRequest = { ...problem, classifier: classifier.name, params: { ...params } }
    const id = ++run.current
    setTrained(request)
    // The extra analyses belong to the previous model now: keep them on screen, mark them stale.
    setImportance((prev) => ({ status: "idle", data: prev.data }))
    setLearning((prev) => ({ status: "idle", data: prev.data }))
    setCv((prev) => ({ status: "idle", data: prev.data }))
    load(setResult, () => api.train(request), () => run.current === id)
  }, [classifier, params, problem])

  const compare = React.useCallback(() => {
    if (!problem) return
    const id = run.current
    load(setComparison, () => api.compare(problem), () => run.current === id)
  }, [problem])

  const ready = result.status === "ready"

  const loadImportance = React.useCallback(
    (retry = false) => {
      if (!trained || !ready) return
      if (importance.status !== "idle" && !(retry && importance.status === "error")) return
      const id = run.current
      load(setImportance, () => api.featureImportance(trained), () => run.current === id)
    },
    [trained, ready, importance.status]
  )

  const loadLearning = React.useCallback(
    (retry = false) => {
      if (!trained || !ready) return
      if (learning.status !== "idle" && !(retry && learning.status === "error")) return
      const id = run.current
      load(setLearning, () => api.learningCurve(trained), () => run.current === id)
    },
    [trained, ready, learning.status]
  )

  const loadCv = React.useCallback(
    (retry = false) => {
      if (!trained || !ready) return
      if (cv.status !== "idle" && !(retry && cv.status === "error")) return
      const id = run.current
      load(setCv, () => api.crossValidation(trained), () => run.current === id)
    },
    [trained, ready, cv.status]
  )

  const dirty =
    trained !== null &&
    classifier !== null &&
    (trained.classifier !== classifier.name || !sameParams(trained.params, params))

  const tabState = React.useMemo<TabState>(() => ({ tab, direction }), [tab, direction])

  // Memoised so that it changes only when something in it does (a tab change is not one).
  const value = React.useMemo<AppState>(() => ({
    setTab,
    getTab,
    meta,
    datasets,
    dataset,
    notice,
    dismissNotice,
    reload,
    chooseDataset,
    chooseTarget,
    choosePositive,
    uploading,
    uploadDataset,
    removeDataset,
    classifier,
    selectClassifier: setClassifierName,
    params,
    setParam,
    resetParams,
    isDefault,
    trained,
    dirty,
    result,
    importance,
    learning,
    cv,
    comparison,
    canTrain,
    train,
    compare,
    loadImportance,
    loadLearning,
    loadCv,
  }), [
    setTab, getTab, meta, datasets, dataset, notice, dismissNotice, reload,
    chooseDataset, chooseTarget, choosePositive, uploading, uploadDataset, removeDataset,
    classifier, params, setParam, resetParams, isDefault, trained, dirty,
    result, importance, learning, cv, comparison, canTrain, train, compare,
    loadImportance, loadLearning, loadCv,
  ])

  return (
    <AppContext.Provider value={value}>
      <TabContext.Provider value={tabState}>{children}</TabContext.Provider>
    </AppContext.Provider>
  )
}

export function useTab() {
  const context = React.useContext(TabContext)
  if (context === undefined) throw new Error("useTab must be used within an AppProvider")
  return context
}

export function useApp() {
  const context = React.useContext(AppContext)
  if (context === undefined) throw new Error("useApp must be used within an AppProvider")
  return context
}
