from __future__ import annotations

from backend.services import encoder_inference, repo


def main():
    if not encoder_inference.is_model_ready():
        raise RuntimeError("Exported model weights are not available in backend/services/model_weights.")

    df = repo.load_articles()
    if df.empty:
        print("no_articles=0")
        return

    updated = 0
    for record in df.fillna("").to_dict(orient="records"):
        vector = encoder_inference.encode_news(record)
        repo.update_article_learned_embedding(record["article_id"], vector)
        updated += 1
        if updated % 100 == 0:
            print(f"updated={updated}")
    print(f"updated={updated}")


if __name__ == "__main__":
    main()
