from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="backend/training/checkpoints/attention_encoders.pt",
    )
    parser.add_argument(
        "--output",
        default="backend/services/model_weights/attention_encoders.pt",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, output)
    print(f"exported={output}")


if __name__ == "__main__":
    main()
