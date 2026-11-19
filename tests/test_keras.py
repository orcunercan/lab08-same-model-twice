"""Tests for the Keras version and for a fair comparison with PyTorch."""

import keras
import numpy as np
import pytest
import torch

from keras_model import build_keras_net, copy_torch_weights
from model import BikeNet
from train import MAIN
from train_keras import train_keras_run

SHORT = {**MAIN, "epochs": 3}


def test_keras_network_returns_one_value_per_row():
    model = build_keras_net(n_features=8)
    out = model.predict(np.zeros((5, 8), dtype="float32"), verbose=0)
    assert out.shape == (5, 1)
    assert out.flatten().shape == (5,)  # flattened, it matches targets of shape (N,)


@pytest.mark.parametrize("hidden", [8, 64, 256])
def test_parameter_count_equals_pytorch(hidden):
    torch_model = BikeNet(n_features=8, hidden=hidden)
    keras_model = build_keras_net(n_features=8, hidden=hidden)
    torch_count = sum(p.numel() for p in torch_model.parameters())
    assert keras_model.count_params() == torch_count


def test_weight_copy_gives_the_same_predictions():
    # Random inputs and a square 64 x 64 hidden layer: if the copy forgot to
    # transpose, that layer would still load but the predictions would differ.
    torch_model = BikeNet(n_features=8)
    keras_model = build_keras_net(n_features=8)
    copy_torch_weights(torch_model, keras_model)
    x = torch.randn(32, 8)
    with torch.no_grad():
        expected = torch_model(x).numpy()
    got = keras_model.predict(x.numpy(), verbose=0).flatten()
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_both_versions_use_the_same_rows(splits):
    # train_keras.py gives Keras splits.train.x.numpy(): the same numbers PyTorch sees
    for part in (splits.train, splits.val, splits.test):
        np.testing.assert_array_equal(part.x.numpy(), part.x.detach().cpu().numpy())
        assert part.x.numpy().dtype == np.float32
        assert part.x.numpy().shape == tuple(part.x.shape)


def test_keras_network_can_memorise_one_small_batch(small_batch):
    x, y = (t.numpy() for t in small_batch)
    keras.utils.set_random_seed(0)
    model = build_keras_net(n_features=x.shape[1])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2), loss="mse")
    start = model.evaluate(x, y, verbose=0)
    model.fit(x, y, epochs=300, batch_size=len(y), verbose=0)
    end = model.evaluate(x, y, verbose=0)
    assert end < 0.05 * start


def test_saved_and_reloaded_model_predicts_the_same(tmp_path, small_batch):
    x = small_batch[0].numpy()
    model = build_keras_net(n_features=x.shape[1])
    path = tmp_path / "model.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)
    np.testing.assert_allclose(
        reloaded.predict(x, verbose=0), model.predict(x, verbose=0), atol=1e-6
    )


def test_best_weights_are_restored(splits, tmp_path):
    model, history = train_keras_run(splits, SHORT, checkpoint=tmp_path / "b.keras")
    val_loss = model.evaluate(splits.val.x.numpy(), splits.val.y.numpy(), verbose=0)[0]
    assert val_loss == pytest.approx(history["val_loss"].min(), rel=1e-4)


def test_the_same_seed_repeats_a_short_run(splits, tmp_path):
    _, first = train_keras_run(splits, SHORT, seed=5, checkpoint=tmp_path / "1.keras")
    _, again = train_keras_run(splits, SHORT, seed=5, checkpoint=tmp_path / "2.keras")
    np.testing.assert_allclose(first["val_loss"], again["val_loss"], rtol=1e-6)
