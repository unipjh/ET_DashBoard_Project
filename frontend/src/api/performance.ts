import { trackEvent } from './logger'

type LayoutShiftEntry = PerformanceEntry & {
  value: number
  hadRecentInput: boolean
}

let lcpMs = 0
let cls = 0
let maxEventMs = 0
let hasReported = false

const supportsObserver = (type: string) =>
  typeof PerformanceObserver !== 'undefined' &&
  PerformanceObserver.supportedEntryTypes?.includes(type)

const reportOnce = () => {
  if (hasReported) return
  hasReported = true

  const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming | undefined
  trackEvent('client_performance', null, {
    path: window.location.pathname,
    lcp_ms: Math.round(lcpMs),
    cls: Number(cls.toFixed(4)),
    max_event_ms: Math.round(maxEventMs),
    dom_content_loaded_ms: navigation ? Math.round(navigation.domContentLoadedEventEnd) : null,
    load_event_ms: navigation ? Math.round(navigation.loadEventEnd) : null,
  })
}

export const initPerformanceMonitoring = () => {
  if (typeof window === 'undefined') return

  if (supportsObserver('largest-contentful-paint')) {
    const observer = new PerformanceObserver((list) => {
      const entries = list.getEntries()
      const lastEntry = entries[entries.length - 1]
      if (lastEntry) lcpMs = lastEntry.startTime
    })
    observer.observe({ type: 'largest-contentful-paint', buffered: true })
  }

  if (supportsObserver('layout-shift')) {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries() as LayoutShiftEntry[]) {
        if (!entry.hadRecentInput) cls += entry.value
      }
    })
    observer.observe({ type: 'layout-shift', buffered: true })
  }

  if (supportsObserver('event')) {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        maxEventMs = Math.max(maxEventMs, entry.duration)
      }
    })
    observer.observe({ type: 'event', buffered: true })
  }

  window.setTimeout(reportOnce, 10000)
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') reportOnce()
  }, { once: true })
}
