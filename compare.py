"""Train the main setup in both frameworks on several seeds: python compare.py"""

import time

import pandas as pd
import torch
from torch import nn

from model import BikeNet
from prepare import SEED, make_loaders, make_splits
from train import MAIN, REPORTS, fit, run_epoch
from train_keras import MODELS, mae, train_keras_run

SEEDS = [SEED, 7, 2026]


def train_torch_run(splits, settings, seed):
    """The Lab 07 PyTorch run, with the seed passed in instead of fixed."""
    torch.manual_seed(seed)
    train_loader, val_loader, _ = make_loaders(
        splits, batch_size=settings["batch_size"], seed=seed
    )
    model = BikeNet(splits.n_features, settings["hidden"], settings["layers"])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    history = fit(
        model, train_loader, val_loader, nn.MSELoss(), optimizer, settings["epochs"]
    )
    return model, pd.DataFrame(history)


def torch_test_mae(model, splits) -> float:
    """Score a PyTorch model on the test rows, once."""
    _, _, test_loader = make_loaders(splits, seed=SEED)
    _, test_mae = run_epoch(model, test_loader, nn.MSELoss())
    return test_mae


def result_row(framework, seed, history, test_mae, seconds) -> dict:
    """One line of comparison.csv, read from the best epoch of the run."""
    best = history.loc[history["val_loss"].idxmin()]
    return {
        "framework": framework,
        "seed": seed,
        "best_epoch": int(best["epoch"]),
        "val_mae": round(float(best["val_mae"]), 3),
        "test_mae": round(test_mae, 3),
        "seconds": round(seconds, 1),
    }


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    splits = make_splits(SEED)  # the same split and rows for every run
    rows = []
    for seed in SEEDS:
        start = time.perf_counter()
        model, history = train_torch_run(splits, MAIN, seed)
        seconds = time.perf_counter() - start
        rows.append(
            result_row("pytorch", seed, history, torch_test_mae(model, splits), seconds)
        )
        print(rows[-1])

        start = time.perf_counter()
        checkpoint = MODELS / f"keras_seed{seed}.keras"
        model, history = train_keras_run(splits, MAIN, seed, checkpoint)
        seconds = time.perf_counter() - start
        rows.append(
            result_row("keras", seed, history, mae(model, splits.test), seconds)
        )
        print(rows[-1])

    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "comparison.csv", index=False)
    summary = table.groupby("framework")[["val_mae", "test_mae", "seconds"]].agg(
        ["mean", "std"]
    )
    print(summary.round(3))


if __name__ == "__main__":
    main()