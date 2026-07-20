"""event_logs/feedback_logs → 학습·평가용 InteractionSample 빌더 (torch 불필요).

dataset.py(학습)와 evaluate_offline.py(오프라인 평가)가 공유한다.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime

from backend.services import repo


POSITIVE_EVENT_TYPES = {
    "view_article_detail",
    "view_detail",
    "click_article",
    "click_related",
    "click_feedback",
}


@dataclass(frozen=True)
class InteractionSample:
    subject_id: str
    history_article_ids: list[str]
    positive_article_id: str
    negative_article_ids: list[str]
    event_at: str = ""


def parse_event_data(raw_value) -> dict:
    if isinstance(raw_value, dict):
        return raw_value
    if not raw_value:
        return {}
    try:
        return json.loads(str(raw_value))
    except Exception:
        return {}


def extract_impression_article_ids(raw_event_data) -> list[str]:
    event_data = parse_event_data(raw_event_data)
    articles = event_data.get("articles") or []
    if not articles and event_data.get("article_ids"):
        articles = event_data.get("article_ids") or []

    article_ids: list[str] = []
    for item in articles:
        article_id = item.get("article_id") if isinstance(item, dict) else item
        if article_id:
            article_ids.append(str(article_id))
    return article_ids


def _subject_id(user_id, session_id=None) -> str:
    """계정/세션 ID 충돌을 피하는 평가용 subject key."""
    if user_id:
        return f"user:{user_id}"
    return f"session:{session_id or 'guest'}"


def _as_datetime(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def build_temporal_samples_from_rows(
    event_rows: list[dict],
    feedback_rows: list[dict],
    history_size: int = 20,
    negative_count: int = 4,
    positive_dedupe_seconds: int = 30,
    allowed_article_ids: set[str] | None = None,
) -> list[InteractionSample]:
    """클릭 시점 이전의 실제 impression만 negative로 사용하는 리플레이 샘플러.

    같은 기사에서 짧은 시간 안에 연속 발생하는 click/view 이벤트는 하나의
    positive로 합친다. 관측 negative가 ``negative_count``보다 부족한 시점은
    catalog fallback으로 채우지 않고 제외하여 미래 기사 혼입을 막는다.
    """
    actions_by_subject: dict[str, list[tuple[str, int, str, object]]] = {}

    for row in event_rows:
        subject = _subject_id(row.get("user_id"), row.get("session_id"))
        created_at = str(row.get("created_at") or "")
        event_type = row.get("event_type")
        if event_type == "impression":
            article_ids = extract_impression_article_ids(row.get("event_data"))
            if allowed_article_ids is not None:
                article_ids = [article_id for article_id in article_ids if article_id in allowed_article_ids]
            if article_ids:
                actions_by_subject.setdefault(subject, []).append(
                    (created_at, 0, "impression", article_ids)
                )
        elif event_type in POSITIVE_EVENT_TYPES and row.get("article_id"):
            if allowed_article_ids is not None and str(row["article_id"]) not in allowed_article_ids:
                continue
            actions_by_subject.setdefault(subject, []).append(
                (created_at, 1, "positive", str(row["article_id"]))
            )

    for row in feedback_rows:
        if row.get("feedback_type") != "like" or not row.get("article_id"):
            continue
        if allowed_article_ids is not None and str(row["article_id"]) not in allowed_article_ids:
            continue
        subject = _subject_id(row.get("user_id"))
        actions_by_subject.setdefault(subject, []).append(
            (str(row.get("created_at") or ""), 1, "positive", str(row["article_id"]))
        )

    samples: list[InteractionSample] = []
    for subject, actions in actions_by_subject.items():
        history: list[str] = []
        history_set: set[str] = set()
        observed_impressions: list[str] = []
        last_positive_at: dict[str, datetime | None] = {}

        for created_at, _, action_type, payload in sorted(actions):
            if action_type == "impression":
                observed_impressions.extend(str(article_id) for article_id in payload)
                continue

            positive_article_id = str(payload)
            current_time = _as_datetime(created_at)
            previous_time = last_positive_at.get(positive_article_id)
            if current_time is not None and previous_time is not None:
                try:
                    if (current_time - previous_time).total_seconds() <= positive_dedupe_seconds:
                        continue
                except TypeError:
                    pass
            last_positive_at[positive_article_id] = current_time

            if history:
                negatives: list[str] = []
                for article_id in reversed(observed_impressions):
                    if (
                        article_id != positive_article_id
                        and article_id not in history_set
                        and article_id not in negatives
                    ):
                        negatives.append(article_id)
                    if len(negatives) >= negative_count:
                        break

                if len(negatives) >= negative_count:
                    samples.append(
                        InteractionSample(
                            subject_id=subject,
                            history_article_ids=history[-history_size:],
                            positive_article_id=positive_article_id,
                            negative_article_ids=negatives[:negative_count],
                            event_at=created_at,
                        )
                    )

            history.append(positive_article_id)
            history_set.add(positive_article_id)

    return sorted(samples, key=lambda sample: sample.event_at)


def build_samples_from_rows(
    event_rows: list[dict],
    feedback_rows: list[dict],
    all_article_ids: list[str],
    history_size: int = 20,
    negative_count: int = 4,
) -> list[InteractionSample]:
    positives_by_subject: dict[str, list[tuple[str, str]]] = {}
    impressions_by_subject: dict[str, list[str]] = {}
    clicked_by_subject: dict[str, set[str]] = {}

    for row in event_rows:
        subject_id = str(row.get("user_id") or row.get("session_id") or "guest")
        event_type = row.get("event_type")
        article_id = row.get("article_id")
        created_at = str(row.get("created_at") or "")
        if event_type == "impression":
            impressions_by_subject.setdefault(subject_id, []).extend(
                extract_impression_article_ids(row.get("event_data"))
            )
        elif event_type in POSITIVE_EVENT_TYPES and article_id:
            positives_by_subject.setdefault(subject_id, []).append((created_at, str(article_id)))
            clicked_by_subject.setdefault(subject_id, set()).add(str(article_id))

    for row in feedback_rows:
        if row.get("feedback_type") != "like":
            continue
        subject_id = str(row.get("user_id") or "guest")
        article_id = row.get("article_id")
        created_at = str(row.get("created_at") or "")
        if article_id:
            positives_by_subject.setdefault(subject_id, []).append((created_at, str(article_id)))
            clicked_by_subject.setdefault(subject_id, set()).add(str(article_id))

    samples: list[InteractionSample] = []
    article_pool = [str(article_id) for article_id in all_article_ids]
    for subject_id, positives in positives_by_subject.items():
        positives = sorted(positives, key=lambda item: item[0])
        history: list[str] = []
        clicked = clicked_by_subject.get(subject_id, set())
        impression_negatives = [
            article_id
            for article_id in impressions_by_subject.get(subject_id, [])
            if article_id not in clicked
        ]

        for event_at, positive_article_id in positives:
            if history:
                negatives = list(dict.fromkeys(impression_negatives))
                if len(negatives) < negative_count:
                    fallback = [
                        article_id
                        for article_id in article_pool
                        if article_id != positive_article_id and article_id not in history
                    ]
                    random.shuffle(fallback)
                    negatives.extend(fallback[: negative_count - len(negatives)])
                samples.append(
                    InteractionSample(
                        subject_id=subject_id,
                        history_article_ids=history[-history_size:],
                        positive_article_id=positive_article_id,
                        negative_article_ids=negatives[:negative_count],
                        event_at=event_at,
                    )
                )
            history.append(positive_article_id)
    return samples


def load_samples_from_db(
    history_size: int = 20,
    negative_count: int = 4,
    sampling_mode: str = "temporal",
) -> list[InteractionSample]:
    con = repo._get_con()
    cur = con.cursor()
    try:
        cur.execute("SELECT article_id, embed_summary FROM articles")
        article_rows = cur.fetchall()
        all_article_ids = [str(row[0]) for row in article_rows]
        embedded_article_ids = {
            str(article_id)
            for article_id, embed_summary in article_rows
            if repo._parse_vector(embed_summary)
        }
        cur.execute(
            """
            SELECT session_id, user_id, event_type, article_id, event_data, created_at
            FROM event_logs
            ORDER BY created_at ASC
            """
        )
        event_rows = [
            {
                "session_id": row[0],
                "user_id": row[1],
                "event_type": row[2],
                "article_id": row[3],
                "event_data": row[4],
                "created_at": row[5],
            }
            for row in cur.fetchall()
        ]
        cur.execute(
            """
            SELECT user_id, article_id, feedback_type, created_at
            FROM feedback_logs
            ORDER BY created_at ASC
            """
        )
        feedback_rows = [
            {
                "user_id": row[0],
                "article_id": row[1],
                "feedback_type": row[2],
                "created_at": row[3],
            }
            for row in cur.fetchall()
        ]
    finally:
        cur.close()
        con.close()

    if sampling_mode == "temporal":
        return build_temporal_samples_from_rows(
            event_rows=event_rows,
            feedback_rows=feedback_rows,
            history_size=history_size,
            negative_count=negative_count,
            allowed_article_ids=embedded_article_ids,
        )
    if sampling_mode == "legacy":
        return build_samples_from_rows(
            event_rows=event_rows,
            feedback_rows=feedback_rows,
            all_article_ids=all_article_ids,
            history_size=history_size,
            negative_count=negative_count,
        )
    raise ValueError(f"지원하지 않는 sampling_mode: {sampling_mode}")
