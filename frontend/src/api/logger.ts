import axios from 'axios';

const getSessionId = () => {
  let sessionId = sessionStorage.getItem('et_session_id');
  if (!sessionId) {
    sessionId = 'sess_' + Math.random().toString(36).substring(2, 11);
    sessionStorage.setItem('et_session_id', sessionId);
  }
  return sessionId;
};

export const trackEvent = (eventType: string, articleId?: string | null, eventData?: any) => {
  const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  const targetUrl = `${baseURL}/api/logs`;
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
