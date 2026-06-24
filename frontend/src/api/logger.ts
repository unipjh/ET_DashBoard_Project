import axios from 'axios';
import { API_BASE_URL } from './client';

export const getSessionId = () => {
  let sessionId = sessionStorage.getItem('et_session_id');
  if (!sessionId) {
    sessionId = 'sess_' + Math.random().toString(36).substring(2, 11);
    sessionStorage.setItem('et_session_id', sessionId);
  }
  return sessionId;
};

export const trackEvent = (eventType: string, articleId?: string | null, eventData?: Record<string, unknown>) => {
  const targetUrl = `${API_BASE_URL}/api/logs`;
  const userId = localStorage.getItem('et_user') || null;

  axios.post(targetUrl, {
    session_id: getSessionId(),
    event_type: eventType,
    article_id: articleId || null,
    event_data: eventData || {},
    user_id: userId,
  }).catch(err => {
     console.error(`❌ [로그 전송 실패] ${eventType}`, err);
  });
};

type ImpressionItem = {
  article_id: string
  position: number
}

const sentImpressionKeys = new Set<string>()

type PendingImpressionBatch = {
  articles: ImpressionItem[]
  eventData: Record<string, unknown>
  timer: number
}

const pendingImpressionBatches = new Map<string, PendingImpressionBatch>()

export const trackImpressions = (
  articles: ImpressionItem[],
  eventData: Record<string, unknown> = {},
) => {
  const contextKey = String(
    eventData.context_key
    || `${eventData.source || 'main'}:${eventData.page || ''}:${eventData.category || ''}:${eventData.query || ''}`
  )
  const freshItems = articles.filter((item) => {
    if (!item.article_id) return false
    const key = `${getSessionId()}:${contextKey}:${item.article_id}:${item.position}`
    if (sentImpressionKeys.has(key)) return false
    sentImpressionKeys.add(key)
    return true
  })

  if (freshItems.length === 0) return

  const pending = pendingImpressionBatches.get(contextKey)
  if (pending) window.clearTimeout(pending.timer)

  const mergedArticles = [...(pending?.articles || []), ...freshItems]
  const dedupedArticles = Array.from(
    new Map(mergedArticles.map((item) => [`${item.article_id}:${item.position}`, item])).values()
  )

  const timer = window.setTimeout(() => {
    const batch = pendingImpressionBatches.get(contextKey)
    if (!batch) return
    pendingImpressionBatches.delete(contextKey)
    trackEvent('impression', null, {
      ...batch.eventData,
      context_key: contextKey,
      articles: batch.articles,
      article_ids: batch.articles.map((item) => item.article_id),
    })
  }, 350)

  pendingImpressionBatches.set(contextKey, {
    articles: dedupedArticles,
    eventData,
    timer,
  })
}
