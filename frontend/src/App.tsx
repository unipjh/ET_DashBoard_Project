import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import MainPage from './pages/MainPage'
import DetailPage from './pages/DetailPage'
import AdminPage from './pages/AdminPage'
import FeedbackPage from './pages/FeedbackPage'
import LogPage from './pages/LogPage'
import Logo from './components/Logo'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60 * 1000,
      gcTime: 5 * 60 * 1000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

function SplashScreen({ onDone }: { onDone: () => void }) {
  const [fadeOut, setFadeOut] = useState(false)

  useEffect(() => {
    const alreadySeen = sessionStorage.getItem('et_splash_seen') === '1'
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const fadeDelay = alreadySeen || reduceMotion ? 120 : 520
    const doneDelay = alreadySeen || reduceMotion ? 180 : 760

    const fadeTimer = setTimeout(() => setFadeOut(true), fadeDelay)
    const doneTimer = setTimeout(() => {
      sessionStorage.setItem('et_splash_seen', '1')
      onDone()
    }, doneDelay)
    return () => {
      clearTimeout(fadeTimer)
      clearTimeout(doneTimer)
    }
  }, [onDone])

  return (
    <div
      className={`fixed inset-0 z-50 flex flex-col items-center justify-center bg-paper transition-opacity duration-500 ease-out ${fadeOut ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}
    >
      <Logo className="splash-logo" />
    </div>
  )
}

export default function App() {
  const [splashDone, setSplashDone] = useState(false)

  return (
    <QueryClientProvider client={queryClient}>
      {!splashDone && <SplashScreen onDone={() => setSplashDone(true)} />}
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<MainPage />} />
          <Route path="/article/:id" element={<DetailPage />} />
          <Route path="/admin" element={<AdminPage />} />
          <Route path="/feedback" element={<FeedbackPage />} />
          <Route path="/log" element={<LogPage />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
