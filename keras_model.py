"""The Keras network, built to match BikeNet, and a copy of PyTorch weights into it."""

import keras
from torch import nn


def build_keras_net(n_features: int, hidden: int = 64, layers: int = 2) -> keras.Model:
    """Hidden Dense layers with ReLU, then one output with no activation."""
    blocks = [keras.Input(shape=(n_features,))]
    for _ in range(layers):
        blocks.append(keras.layers.Dense(hidden, activation="relu"))
    blocks.append(keras.layers.Dense(1))
    return keras.Sequential(blocks)


def copy_torch_weights(torch_model: nn.Module, keras_model: keras.Model) -> None:
    """Copy every nn.Linear of the PyTorch network into the matching Dense layer.

    PyTorch stores a weight as (outputs, inputs) and Keras stores a kernel as
    (inputs, outputs), so each weight is transposed on the way across.
    """
    linears = [m for m in torch_model.modules() if isinstance(m, nn.Linear)]
    denses = [
        layer for layer in keras_model.layers if isinstance(layer, keras.layers.Dense)
    ]
    if len(linears) != len(denses):
        raise ValueError(f"{len(linears)} Linear layers but {len(denses)} Dense layers")
    for linear, dense in zip(linears, denses, strict=True):
        kernel = linear.weight.detach().numpy().T
        bias = linear.bias.detach().numpy()
        dense.set_weights([kernel, bias])