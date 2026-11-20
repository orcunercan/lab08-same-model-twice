# PyTorch or Keras for the weekly bike demand job

## What I compared

The same network (two hidden layers of 64 with ReLU, one output, 4,801 parameters) was built in
both frameworks. Before training, I copied the PyTorch weights into the Keras model and the two
predicted the same values on the validation rows (largest difference 3e-08), so they compute the
same function. Both versions then trained on the same rows with MSE, Adam at 0.003, batches of 64
and 150 epochs, on three seeds (42, 7, 2026). The settings I could not make identical are the
initial weights (Keras uses Glorot uniform, PyTorch its own default) and the order in which each
framework shuffles the batches, so a seed repeats a run inside one framework but not across them.

## What I observed

1. **Results.** PyTorch: validation MAE 2.626 ± 0.009 and test MAE 2.484 ± 0.030. Keras: 2.727 ±
   0.115 and 2.611 ± 0.143. The averages are close, but Keras varied about twelve times more
   between seeds (from 2.534 to 2.776 test MAE), while PyTorch stayed between 2.461 and 2.518.
2. **Time.** On my laptop CPU a PyTorch run took 17.5 seconds on average and a Keras run 35.5, so
   Keras was about twice as slow for this small model.
3. **Code.** The PyTorch loop (`run_epoch` and `fit`) is 38 lines I had to write and test by hand,
   including clearing gradients and putting back the best weights. In Keras the loop is one `fit`
   call, and two callbacks do the best-weights and checkpoint work.
4. **Shapes and saving.** Keras `predict` returns shape (N, 1), so every score needs `.flatten()`.
   The saved `.keras` file is about 83 KB and holds the architecture and optimizer as well, while
   the PyTorch state dict is about 22 KB but needs the `BikeNet` class in code to load.

## Recommendation: PyTorch

For a weekly job run by two people, the steadier results and the faster runs matter more than the
shorter code, and the loop is already written and covered by tests. Most pretrained models on
Hugging Face are published for PyTorch, so a later move to them starts from the same framework.

## What we give up

The team member who knows Keras has to learn PyTorch, and we keep maintaining our own loop
instead of relying on `fit` and its callbacks. A saved PyTorch model also needs our code to load.

## What would change my mind

If the team kept only the Keras person, or if the model had to be served with TensorFlow tools,
Keras would be the better choice, since its single `.keras` file loads without our code.