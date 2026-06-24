from __future__ import annotations

import subprocess
import sys
from shutil import which
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NPM = which("npm.cmd") or which("npm") or "npm"


PLAYWRIGHT = which("playwright") or (ROOT / "frontend" / "node_modules" / ".bin" / "playwright.cmd")

CHECKS = [
    ("frontend lint", [NPM, "run", "lint"], ROOT / "frontend"),
    ("frontend build", [NPM, "run", "build"], ROOT / "frontend"),
    ("frontend e2e", [str(PLAYWRIGHT), "test", "--reporter=list"], ROOT / "frontend"),
    ("backend import", [sys.executable, "-c", "from backend.main import app; print(app.title)"], ROOT),
    ("backend smoke tests", [sys.executable, "-m", "unittest", "backend.tests.test_smoke"], ROOT),
]


def run_check(name: str, command: list[str], cwd: Path) -> bool:
    print(f"\n== {name} ==")
    result = subprocess.run(command, cwd=cwd, text=True)
    if result.returncode == 0:
        print(f"PASS: {name}")
        return True
    print(f"FAIL: {name} (exit {result.returncode})")
    return False


def main() -> int:
    failures = [
        name
        for name, command, cwd in CHECKS
        if not run_check(name, command, cwd)
    ]
    if failures:
        print("\nPreflight failed:")
        for name in failures:
            print(f"- {name}")
        return 1

    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
