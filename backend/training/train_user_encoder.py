from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from backend.training.dataset import (
    RecommendationDataset,
    collate_recommendation_batch,
    load_article_features_from_db,
    load_samples_from_db,
)
from backend.training.news_encoder import NewsEncoder
from backend.training.user_encoder import UserEncoder


def encode_feature_batch(news_encoder: NewsEncoder, title, summary, category):
    original_shape = title.shape[:-1]
    flat_title = title.reshape(-1, title.shape[-1])
    flat_summary = summary.reshape(-1, summary.shape[-1])
    flat_category = category.reshape(-1, category.shape[-1])
    encoded = news_encoder(flat_title, flat_summary, flat_category)
    return encoded.reshape(*original_shape, encoded.shape[-1])


def score_batch(news_encoder: NewsEncoder, user_encoder: UserEncoder, batch: dict):
    history_vectors = encode_feature_batch(
        news_encoder,
        batch["history_title"],
        batch["history_summary"],
        batch["history_category"],
    )
    candidate_vectors = encode_feature_batch(
        news_encoder,
        batch["candidate_title"],
        batch["candidate_summary"],
        batch["candidate_category"],
    )
    user_vectors = user_encoder(history_vectors, batch["history_mask"])
    return torch.bmm(candidate_vectors, user_vectors.unsqueeze(-1)).squeeze(-1)


def evaluate(news_encoder: NewsEncoder, user_encoder: UserEncoder, loader: DataLoader):
    news_encoder.eval()
    user_encoder.eval()
    auc_hits = 0
    auc_total = 0
    recall_hits = 0
    total = 0
    baseline_auc_hits = 0
    with torch.no_grad():
        for batch in loader:
            scores = score_batch(news_encoder, user_encoder, batch)
            positive_scores = scores[:, 0]
            negative_scores = scores[:, 1:]
            auc_hits += (positive_scores.unsqueeze(1) > negative_scores).sum().item()
            auc_total += negative_scores.numel()
            recall_hits += (scores.argsort(dim=1, descending=True)[:, :10] == 0).any(dim=1).sum().item()
            total += scores.shape[0]

            baseline_user = torch.nn.functional.normalize(
                batch["history_summary"].mean(dim=1),
                p=2,
                dim=-1,
                eps=1e-12,
            )
            baseline_candidates = torch.nn.functional.normalize(
                batch["candidate_summary"],
                p=2,
                dim=-1,
                eps=1e-12,
            )
            baseline_scores = torch.bmm(baseline_candidates, baseline_user.unsqueeze(-1)).squeeze(-1)
            baseline_auc_hits += (
                baseline_scores[:, 0].unsqueeze(1) > baseline_scores[:, 1:]
            ).sum().item()

    return {
        "auc": auc_hits / auc_total if auc_total else 0.0,
        "recall_at_10": recall_hits / total if total else 0.0,
        "baseline_auc": baseline_auc_hits / auc_total if auc_total else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--negative-count", type=int, default=4)
    parser.add_argument("--history-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--checkpoint-dir", default="backend/training/checkpoints")
    parser.add_argument("--synthetic", action="store_true",
                        help="실제 event_logs 대신 합성 데이터로 학습")
    parser.add_argument("--synthetic-sessions", type=int, default=80,
                        help="합성 데이터: 페르소나당 세션 수 (기본 80)")
    parser.add_argument("--synthetic-seed", type=int, default=42,
                        help="합성 데이터 생성 시드 (기본 42)")
    args = parser.parse_args()

    article_features, category_vocab = load_article_features_from_db()

    if args.synthetic:
        from backend.training.generate_synthetic_samples import generate_synthetic_samples
        print(f"합성 데이터 생성 중 (페르소나당 {args.synthetic_sessions}세션, seed={args.synthetic_seed})...")
        samples = generate_synthetic_samples(
            article_features,
            sessions_per_persona=args.synthetic_sessions,
            history_size=args.history_size,
            negative_count=args.negative_count,
            seed=args.synthetic_seed,
        )
    else:
        samples = load_samples_from_db(
            history_size=args.history_size,
            negative_count=args.negative_count,
        )

    samples = [
        sample
        for sample in samples
        if sample.positive_article_id in article_features
        and all(article_id in article_features for article_id in sample.history_article_ids)
        and all(article_id in article_features for article_id in sample.negative_article_ids)
    ]
    if len(samples) < 2:
        if args.synthetic:
            raise RuntimeError("합성 샘플 필터링 후 샘플 부족. DB에 embed_summary가 있는 기사가 충분한지 확인하세요.")
        raise RuntimeError("Not enough interaction samples. Collect Phase 1 impressions and clicks first.")

    split_index = max(1, int(len(samples) * (1 - args.validation_ratio)))
    train_samples = samples[:split_index]
    val_samples = samples[split_index:] or samples[-1:]
    random.shuffle(train_samples)

    train_dataset = RecommendationDataset(train_samples, article_features, category_vocab, args.history_size)
    val_dataset = RecommendationDataset(val_samples, article_features, category_vocab, args.history_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_recommendation_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_recommendation_batch,
    )

    news_encoder = NewsEncoder(category_dim=len(category_vocab))
    user_encoder = UserEncoder(max_history=args.history_size)
    optimizer = torch.optim.AdamW(
        list(news_encoder.parameters()) + list(user_encoder.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    best_auc = -1.0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "attention_encoders.pt"

    for epoch in range(1, args.epochs + 1):
        news_encoder.train()
        user_encoder.train()
        total_loss = 0.0
        total_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            scores = score_batch(news_encoder, user_encoder, batch)
            loss = criterion(scores, batch["label"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1

        metrics = evaluate(news_encoder, user_encoder, val_loader)
        avg_loss = total_loss / max(total_batches, 1)
        print(
            f"epoch={epoch} loss={avg_loss:.4f} "
            f"val_auc={metrics['auc']:.4f} recall@10={metrics['recall_at_10']:.4f} "
            f"baseline_auc={metrics['baseline_auc']:.4f}"
        )
        if metrics["auc"] >= best_auc:
            best_auc = metrics["auc"]
            torch.save(
                {
                    "news_encoder": news_encoder.state_dict(),
                    "user_encoder": user_encoder.state_dict(),
                    "category_vocab": category_vocab,
                    "config": {
                        "embedding_dim": 768,
                        "history_size": args.history_size,
                        "category_dim": len(category_vocab),
                    },
                    "metrics": metrics,
                },
                checkpoint_path,
            )

    print(f"saved_checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()

