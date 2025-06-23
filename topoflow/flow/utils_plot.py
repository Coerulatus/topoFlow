"""Utility functions for plotting results from a model."""

import os
import torch
import matplotlib.pyplot as plt


def plot_results(results, max_cols=4, title="Sampling Results", path=None, clamp=False):
    """Plot results from a model.

    Args:
        results (list of torch.Tensor): List of tensors to plot. Each tensor can be 2D or 4D.
        max_cols (int): Maximum number of columns in the plot grid.
        title (str): Title for the plots.
        path (str, optional): If provided, save the plot to this path.
        clamp (bool): If True, clamp the output values between -1 and 1.
    """
    if clamp:
        results = [torch.clamp(output, -1, 1) for output in results]
    n_plots = len(results)
    rows = n_plots // max_cols + (n_plots % max_cols > 0)
    cols = min(max_cols, n_plots)
    plt.figure(figsize=(15, 5 * rows))
    fig = plt.gcf()
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    if len(results[0].shape) == 2:
        for i, output in enumerate(results):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.scatter(output[:, 0], output[:, 1], s=1)
            ax.set_title(title + f" {i + 1}")
    elif len(results[0].shape) == 4:
        # if batch dim is a square
        num_tiles = results[0].shape[0]
        side_length = int(num_tiles**0.5)
        if side_length * side_length != num_tiles:
            raise ValueError(f"Unsupported output shape {results[0].shape}.")
        for i, output in enumerate(results):
            mats = [results[i][t] for t in range(num_tiles)]
            rows_ = []
            for j in range(side_length):
                row = torch.cat(mats[j * side_length : (j + 1) * side_length], dim=2)
                rows_.append(row)
            output = torch.cat(rows_, dim=1)
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(output[0].cpu().numpy(), cmap="gray")
            ax.set_title(title + f" {i + 1}")
    else:
        raise ValueError(f"Unsupported output shape {results[0].shape}.")
    plt.tight_layout()
    if path is not None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        plt.savefig(path)
    plt.show()
