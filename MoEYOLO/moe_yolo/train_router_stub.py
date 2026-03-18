from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_feature_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load training rows from jsonl.

    Each line should include:
    {
      "features": [f1, f2, ...],
      "label": 0 or 1
    }
    """
    xs: list[list[float]] = []
    ys: list[int] = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj: dict[str, Any] = json.loads(line)
        xs.append([float(v) for v in obj["features"]])
        ys.append(int(obj["label"]))

    if not xs:
        raise ValueError(f"No usable samples found in {path}")

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic_regression(x: np.ndarray, y: np.ndarray, lr: float, epochs: int) -> tuple[np.ndarray, float]:
    """Tiny baseline trainer with numpy only.

    This provides a reproducible baseline before replacing it with torch training.
    """
    n, d = x.shape
    w: np.ndarray = np.zeros((d,), dtype=np.float32)
    b: float = 0.0

    for _ in range(epochs):
        z: np.ndarray = x @ w + b
        p: np.ndarray = sigmoid(z)

        grad_w: np.ndarray = (x.T @ (p - y)) / n
        grad_b: float = float(np.mean(p - y))

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def evaluate(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> dict[str, Any]:
    p: np.ndarray = sigmoid(x @ w + b)
    pred: np.ndarray = (p >= 0.5).astype(np.float32)
    acc: float = float(np.mean(pred == y))
    return {"accuracy": acc, "samples": int(len(y))}


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Train a simple trigger router baseline.")
    parser.add_argument("--train", type=Path, required=True, help="Path to train jsonl.")
    parser.add_argument("--val", type=Path, help="Optional path to validation jsonl.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=Path("router_weights.npz"))
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()

    x_train: np.ndarray
    y_train: np.ndarray
    x_train, y_train = load_feature_rows(args.train)
    w: np.ndarray
    b: float
    w, b = train_logistic_regression(x_train, y_train, lr=args.lr, epochs=args.epochs)

    train_metrics: dict[str, Any] = evaluate(x_train, y_train, w, b)
    print("train_metrics", json.dumps(train_metrics, ensure_ascii=True))

    if args.val:
        x_val: np.ndarray
        y_val: np.ndarray
        x_val, y_val = load_feature_rows(args.val)
        val_metrics: dict[str, Any] = evaluate(x_val, y_val, w, b)
        print("val_metrics", json.dumps(val_metrics, ensure_ascii=True))

    np.savez(args.out, w=w, b=np.asarray([b], dtype=np.float32))
    print(f"saved router baseline to: {args.out}")


if __name__ == "__main__":
    main()
