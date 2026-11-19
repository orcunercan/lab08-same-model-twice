"""Train the same network in Keras with compile and fit: python train_keras.py"""

from pathlib import Path

import keras
import pandas as pd

from keras_model import build_keras_net
from prepare import SEED, make_splits
from train import MAIN, REPORTS, plot_curves

MODELS = Path("models")


def train_keras_run(
    splits, settings, seed=SEED, checkpoint=MODELS / "keras_best.keras"
):
    """Train one Keras network with the same settings and rows as the PyTorch run."""
    keras.utils.set_random_seed(seed)  # seeds Python, NumPy and TensorFlow at once
    model = build_keras_net(splits.n_features, settings["hidden"], settings["layers"])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=settings["lr"]),
        loss="mse",
        metrics=["mae"],
    )
    checkpoint.parent.mkdir(exist_ok=True)
    callbacks = [
        # patience equal to the epochs: never stop early, only put back the best weights
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=settings["epochs"],
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_loss", save_best_only=True
        ),
    ]
    fitted = model.fit(
        splits.train.x.numpy(),  # the same prepared rows as PyTorch, as NumPy arrays
        splits.train.y.numpy(),
        validation_data=(splits.val.x.numpy(), splits.val.y.numpy()),
        epochs=settings["epochs"],
        batch_size=settings["batch_size"],
        callbacks=callbacks,
        verbose=0,
    )
    log = fitted.history
    history = pd.DataFrame(
        {
            "epoch": range(1, len(log["loss"]) + 1),
            "train_loss": log["loss"],
            "train_mae": log["mae"],
            "val_loss": log["val_loss"],
            "val_mae": log["val_mae"],
        }
    )
    return model, history


def mae(model, part) -> float:
    """Mean absolute error in rentals. predict gives (N, 1), so flatten it first."""
    predictions = model.predict(part.x.numpy(), verbose=0).flatten()
    return float(abs(predictions - part.y.numpy()).mean())


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    splits = make_splits(SEED)
    model, history = train_keras_run(splits, MAIN)
    history.round(4).to_csv(REPORTS / "history_keras.csv", index=False)
    plot_curves(history, "Keras main run", REPORTS / "learning_curves_keras.png")
    best = history.loc[history["val_loss"].idxmin()]
    print(f"best epoch {int(best['epoch'])}, best val loss {best['val_loss']:.3f}")
    print(f"validation MAE of the restored model: {mae(model, splits.val):.3f}")


if __name__ == "__main__":
    main()