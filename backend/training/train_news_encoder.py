from __future__ import annotations

from backend.training.train_user_encoder import main


if __name__ == "__main__":
    print("NewsEncoder is trained end-to-end with UserEncoder. Delegating to train_user_encoder.py.")
    main()

