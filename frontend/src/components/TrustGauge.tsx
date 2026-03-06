interface Criterion {
  score: number
  reason: string
}

interface PerCriteria {
  source_credibility?: Criterion
  evidence_support?: Criterion
  style_neutrality?: Criterion
  logical_consistency?: Criterion
  clickbait_risk?: Criterion
}

const criteriaLabel: Record<string, string> = {
  source_credibility: '출처 신뢰성',
  evidence_support: '근거 지지도',
  style_neutrality: '문체 중립성',
  logical_consistency: '논리 일관성',
  clickbait_risk: '어뷰징 위험도',
}

function scoreColor(score: number, key: string) {
  if (key === 'clickbait_risk') {
    return score >= 7 ? 'bg-red-400' : score >= 4 ? 'bg-yellow-400' : 'bg-green-400'
  }
  return score >= 7 ? 'bg-green-400' : score >= 4 ? 'bg-yellow-400' : 'bg-red-400'
}

export default function TrustGauge({
  score,
  verdict,
  reason,
  perCriteriaJson,
}: {
  score: number
  verdict: string
  reason: string
  perCriteriaJson: string
}) {
  let perCriteria: PerCriteria = {}
  try {
    perCriteria = perCriteriaJson ? JSON.parse(perCriteriaJson) : {}
  } catch {
    perCriteria = {}
  }

  const verdictColor =
    verdict === 'likely_true'
      ? 'text-green-600'
      : verdict === 'likely_false'
      ? 'text-red-600'
      : 'text-yellow-600'

  const verdictLabel =
    verdict === 'likely_true' ? '신뢰 가능' : verdict === 'likely_false' ? '신뢰 불가' : '불확실'

  return (
    <div className="bg-slate-50 rounded-xl border border-slate-200 p-5 space-y-4">
      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm font-semibold text-slate-700">신뢰도 분석</span>
          <span className={`text-sm font-bold ${verdictColor}`}>
            {verdictLabel} · {score}점
          </span>
        </div>
        <div className="w-full bg-slate-200 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all ${score >= 70 ? 'bg-green-500' : score >= 40 ? 'bg-yellow-500' : 'bg-red-500'}`}
            style={{ width: `${score}%` }}
          />
        </div>
      </div>

      {Object.keys(perCriteria).length > 0 && (
        <div className="space-y-2">
          {Object.entries(perCriteria).map(([key, val]) => (
            <div key={key}>
              <div className="flex items-center justify-between text-xs mb-0.5">
                <span className="text-slate-600">{criteriaLabel[key] ?? key}</span>
                <span className="font-medium text-slate-700">{val.score}/10</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-1.5">
                <div
                  className={`h-1.5 rounded-full ${scoreColor(val.score, key)}`}
                  style={{ width: `${val.score * 10}%` }}
                />
              </div>
              {val.reason && <p className="text-xs text-slate-400 mt-0.5">{val.reason}</p>}
            </div>
          ))}
        </div>
      )}

      {reason && (
        <p className="text-xs text-slate-600 border-t border-slate-200 pt-3">{reason}</p>
      )}
    </div>
  )
}
