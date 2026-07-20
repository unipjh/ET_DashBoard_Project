import { test, expect } from '@playwright/test'

const FAKE_ARTICLE = {
  article_id: 'test-001',
  title: '테스트 기사 제목',
  source: '연합뉴스',
  url: 'https://example.com',
  published_at: '2026-06-24',
  summary_text: '테스트 요약입니다.',
  keywords: '[]',
  trust_score: 80,
  trust_verdict: 'likely_true',
  category: 'IT/과학',
}

test.beforeEach(async ({ page }) => {
  await page.route('**/api/logs', route => route.fulfill({ status: 200, body: '{}' }))
  await page.route('**/api/recommendations**', route =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '[]' })
  )
})

test('메인 화면 — 기사 카드 표시', async ({ page }) => {
  await page.route('**/api/articles**', route =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ articles: [FAKE_ARTICLE], total_count: 1 }),
    })
  )
  await page.goto('/')
  await expect(page.getByText('테스트 기사 제목')).toBeVisible()
})

test('상세 페이지 — 기사 제목과 요약 표시', async ({ page }) => {
  await page.route('**/api/articles**', async route => {
    const url = route.request().url()
    if (url.includes('/related')) {
      await route.fulfill({ status: 200, contentType: 'application/json', body: '[]' })
    } else if (/\/api\/articles\/[^/?]+$/.test(url)) {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(FAKE_ARTICLE) })
    } else {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ articles: [FAKE_ARTICLE], total_count: 1 }),
      })
    }
  })
  await page.goto('/article/test-001')
  await expect(page.getByText('테스트 기사 제목')).toBeVisible()
  await expect(page.getByText('테스트 요약입니다.')).toBeVisible()
})

test('어드민 로그인 — 틀린 비밀번호 오류 표시', async ({ page }) => {
  await page.route('**/api/admin/session', route =>
    route.fulfill({
      status: 401,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'Admin authentication failed.' }),
    })
  )
  await page.goto('/admin')
  await page.fill('input[placeholder="관리자 비밀번호"]', 'wrongpassword')
  await page.getByRole('button', { name: '관리자 화면 열기' }).click()
  await expect(page.getByText('비밀번호가 맞지 않습니다.')).toBeVisible()
})

test('빈 상태 — 기사 없음 메시지 표시', async ({ page }) => {
  await page.route('**/api/articles**', route =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ articles: [], total_count: 0 }),
    })
  )
  await page.goto('/')
  await expect(page.getByText('아직 표시할 기사가 없습니다.')).toBeVisible()
})

test('오류 상태 — 서버 에러 메시지 표시', async ({ page }) => {
  await page.route('**/api/articles**', route =>
    route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'DB unavailable' }) })
  )
  await page.goto('/')
  await expect(page.getByText('기사를 불러오지 못했습니다.')).toBeVisible()
})
