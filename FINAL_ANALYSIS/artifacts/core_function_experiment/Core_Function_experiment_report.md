# ET Core Function Experiment Report

- 실행 시각(UTC): 2026-07-11T07:29:16.251271+00:00

- Git commit: 1016f99d

- Worktree dirty: True

- Evaluation code SHA-256: 62c1937d95fa7c7ef47e5d4cd605931908c9bb9ce123539eaac30e6796a3ef5b

- RUN_LIVE_TRUST: False

- RUN_DB_RECOMMENDATION: False

## 결과 사용 판정
| result_scope | status | basis |
| --- | --- | --- |
| 신뢰도 산식·경계 구현 | 사용 가능 — 구현 검증 | 단위검증 9/9 통과 |
| 신뢰도 모델 성능 | 조건부 사용 가능 | Live 표본 30건; 30건 미만 또는 미실행이면 성능 결론 금지 |
| 추천 로직 회귀 | 사용 가능 — 합성 구현 검증 | 합성 표본 12건, 후처리 4/4 통과 |
| 운영 추천 품질 | 보류 | 시간 순서를 보존한 DB 리플레이 100건 이상이 필요함 |

## 핵심 해석
- 평가는 미커밋 변경이 있는 워크트리에서 실행됐다. 재현 시 commit뿐 아니라 evaluation_code_sha256=62c1937d95fa7c7ef47e5d4cd605931908c9bb9ce123539eaac30e6796a3ef5b을 함께 확인해야 한다.
- 현재 로컬 실행 환경의 Attention 경로는 사용 가능 상태다. 이 결과는 Render 준비 상태를 의미하지 않는다.
- 현재 운영 가중치 W2의 캐시 불변쌍 통과율은 100.0%이며 실패 사례는 -다.
- 불변쌍 통과율만으로 가중치 우열을 정할 수 없어, SNU 팩트체크 상관과 AUC가 가장 높은 W2를 선택했다.
- Live 신뢰도 평가 30건 중 실패율은 0.0%다.
- 신뢰도 성능은 저장된 SNU Live criterion 결과를 현재 W2로 재가중한 값이며 추가 API 호출은 없었다.
- 외부 검색 교차검증 5-fold OOF에서 Spearman은 0.391→0.462, 극단 AUC는 0.775→0.800로 개선됐다. 전체 30건 alpha=0.2 수치는 진단값이며 독립 검증값이 아니다.
- DB 리플레이가 비활성 상태이므로 운영 추천 품질 결론은 보류한다.
- 합성 hard-negative 회귀에서 profile MRR=1.000, category MRR=0.333이다. 이는 구현 경로 구분용이며 운영 성능 수치로 인용하지 않는다.
- 인간 직관 페르소나 5-seed에서 hybrid NDCG@3 상대 개선은 20.6%, MRR 개선은 14.3%다.
- DB 리플레이는 클릭 시점 이전 impression만 사용하고 30초 내 중복 positive 및 임베딩 없는 기사를 제거한 temporal sampler를 사용한다.
- 프로필 벡터 지연시간은 DB 검색·네트워크·Attention 추론을 제외한 로컬 microbenchmark다.
- profile 단계는 raw embed_summary만 사용하며 raw chunk embedding 후보와 같은 공간에서 검색한다. Attention 경로의 learned_embedding은 별도 공간으로 분리한다.

## 런타임 준비도
| check | value |
| --- | --- |
| project_root | C:\Users\user\Desktop\ET_by_claude |
| git_commit | 1016f99d |
| worktree_dirty | True |
| changed_or_untracked_path_count | 25 |
| evaluation_code_sha256 | 62c1937d95fa7c7ef47e5d4cd605931908c9bb9ce123539eaac30e6796a3ef5b |
| python | 3.11.9 |
| python_executable | C:\Users\user\Desktop\ET_by_claude\etvenv1\Scripts\python.exe |
| running_in_etvenv1 | True |
| trust_cache_modified_utc | 2026-07-11T01:44:04.224413+00:00 |
| attention_checkpoint_exists | True |
| attention_checkpoint_mb | 9.8 |
| attention_model_ready_here | True |
| profile_embedding_dim | 768 |

## 신뢰도 산식 회귀검증
| test | actual | expected | pass |
| --- | --- | --- | --- |
| maximum_without_clickbait | 100 | 100 | True |
| verdict_true_boundary | likely_true | likely_true | True |
| verdict_uncertain_boundary | uncertain | uncertain | True |
| verdict_false_boundary | likely_false | likely_false | True |
| known_source_rule | 10 | 10 | True |
| unknown_source_neutral | 5 | 5 | True |
| all_yes_predicate | 10 | 10 | True |
| all_uncertain_predicate | 3 | 3 | True |
| cross_penalty_and_evidence_cap | 0 | 0 | True |

## 신뢰도 불변쌍 가중치 비교
| weights | is_current | cases | passed | pass_rate | failed_cases | mean_pair_margin | min_pair_margin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| W2 | True | 9 | 9 | 1.0 | - | 30.5 | 5.0 |
| W0 | False | 9 | 9 | 1.0 | - | 29.0 | 6.0 |
| W1 | False | 9 | 9 | 1.0 | - | 29.75 | 5.0 |
| W3 | False | 9 | 9 | 1.0 | - | 28.875 | 6.0 |
| W4 | False | 9 | 9 | 1.0 | - | 29.5 | 3.0 |

## 신뢰도 Live 평가
| n_valid | n_extreme | accuracy | precision | recall | f1 | roc_auc | spearman_rho | high_label_mean | low_label_mean | tp | tn | fp | fn | requested | failure_rate | latency_p50_sec | latency_p95_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 20 | 0.8 | 0.75 | 0.9 | 0.8181818181818182 | 0.775 | 0.3914081907262832 | 82.3 | 66.6 | 9 | 7 | 3 | 1 | 30 | 0.0 |  |  |

## 외부 검색 교차검증
| n | base_spearman | oof_spearman | base_extreme_auc | oof_extreme_auc | mean_selected_alpha | diagnostic_alpha_0.2_spearman | diagnostic_alpha_0.2_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 0.3914081907262832 | 0.46210430957101584 | 0.775 | 0.8 | 0.27999999999999997 | 0.552088566795546 | 0.86 |

## 추천 합성 랭킹
| strategy | n | HitRate@3 | MRR | NDCG@3 |
| --- | --- | --- | --- | --- |
| profile | 12 | 1.0 | 1.0 | 1.0 |
| hybrid | 12 | 1.0 | 0.75 | 0.8154648767857289 |
| category | 12 | 1.0 | 0.3333333333333333 | 0.5 |
| latest | 12 | 0.0 | 0.16666666666666666 | 0.0 |
| random | 12 | 0.4166666666666667 | 0.3347222222222222 | 0.271821625595243 |

## 추천 후처리 회귀검증
| test | actual | expected | pass |
| --- | --- | --- | --- |
| profile_dimension_768 | 768 | 768 | True |
| recent_item_has_more_weight | True | True | True |
| category_share_at_most_60pct | 0.6 | <=0.6 | True |
| low_trust_demoted | low | low | True |

## 추천 로컬 효율
| operation | runs | latency_p50_ms | latency_p95_ms | peak_python_memory_mb |
| --- | --- | --- | --- | --- |
| build_profile_vector(history=20, dim=768) | 100 | 11.911199995665811 | 16.492199996719137 | 0.1630868911743164 |

## 추천 DB 준비도
_결과 없음_

## 추천 DB 리플레이
| status |
| --- |
| skipped_by_RUN_DB_RECOMMENDATION |

## 인간 직관 페르소나 요약
| seeds | validation_n_total | semantic_ndcg3 | hybrid_ndcg3 | ndcg_relative_gain | semantic_mrr | hybrid_mrr | mrr_relative_gain | low_trust_relative_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 2640 | 0.5231056930993019 | 0.6310379573566083 | 0.206329744984858 | 0.5420075757575751 | 0.6193244949494936 | 0.14264914855452182 | -0.35964912280701755 |

## 페르소나 seed 상세
| seed | validation_n | semantic_ndcg3 | hybrid_ndcg3 | semantic_mrr | hybrid_mrr | semantic_low_trust | hybrid_low_trust |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 528 | 0.5209860618844753 | 0.6169462578125063 | 0.5417929292929285 | 0.608585858585857 | 0.017676767676767676 | 0.010732323232323232 |
| 43 | 528 | 0.5223391328666178 | 0.6275659751217595 | 0.5437815656565649 | 0.6172664141414128 | 0.015782828282828284 | 0.010732323232323232 |
| 44 | 528 | 0.5157103449878301 | 0.6278588705154284 | 0.5344065656565652 | 0.6164772727272715 | 0.013257575757575758 | 0.008838383838383838 |
| 45 | 528 | 0.525428014975655 | 0.6364265201636972 | 0.5413825757575753 | 0.6244318181818169 | 0.013888888888888888 | 0.008838383838383838 |
| 46 | 528 | 0.5310649107819319 | 0.6463921631696498 | 0.5486742424242416 | 0.6298611111111099 | 0.011363636363636364 | 0.006944444444444444 |
