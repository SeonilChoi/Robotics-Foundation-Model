import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import Any


def fill_axes(images: Any, row: int, col: int, path: str, epoch: int):
    fig = plt.figure(figsize=(col*2, row*2))
    for i in range(row * col):
        ax = fig.add_subplot(row, col, i+1)
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig

def save_image(fig: Any, path: str):
    fig.savefig(path)
    plt.close(fig)

def save_progress_image(images: Any, row: int, col: int, path: str, epoch: int):
    fig = fill_axes(images, row, col, path, epoch)
    fig.suptitle(f"Epoch {epoch}")
    save_image(fig, path)

def save_gan_test_image(images: Any, row: int, col: int, path: str, epoch: int):
    fig = fill_axes(images, row, col, path, epoch)
    
    axes = np.array(fig.axes).reshape(row, col)
    left = axes[-1, 0].get_position().x0
    right = axes[-1, -1].get_position().x1
    bottom = axes[-1, 0].get_position().y0
    stride = int(epoch / col)
    
    axis = fig.add_axes([left, bottom - 0.04, right - left, 0.04])
    axis.set_xlim(0, 1)
    axis.set_xticks([0, 1], [0, epoch])
    axis.set_xlabel("Epochs")

    axis.set_yticks([])
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)
    save_image(fig, path)

def save_vae_test_image(images: Any, row: int, col: int, path: str, epoch: int):
    fig = fill_axes(images, row, col, path, epoch)

    axes = np.array(fig.axes).reshape(row, col)
    left = axes[-1, 0].get_position().x0
    right = axes[-1, -1].get_position().x1
    top = axes[0, 0].get_position().y1
    bottom = axes[-1, 0].get_position().y0

    x_axis = fig.add_axes([left, bottom - 0.08, right - left, 0.08])
    y_axis = fig.add_axes([left - 0.08, bottom, 0.08, top - bottom])

    x = np.linspace(-5, 5, 100)
    z1 = stats.norm(0, 1).pdf(x)
    z2 = stats.norm(0, 1).pdf(x)

    x_axis.plot(x, -z1, color="orange", linewidth=10)
    x_axis.fill_between(x, -z1, color="orange", alpha=0.5)
    
    y_axis.plot(-z2, x, color="green", linewidth=10)
    y_axis.fill_betweenx(x, -z2, color="green", alpha=0.5)

    x_axis.spines["left"].set_visible(False)
    x_axis.spines["right"].set_visible(False)
    x_axis.spines["top"].set_visible(False)
    x_axis.spines["bottom"].set_visible(False)
    x_axis.set_xticks([])
    x_axis.set_yticks([])
    x_axis.set_xlabel("$z_1$", fontsize=36)

    y_axis.spines["left"].set_visible(False)
    y_axis.spines["right"].set_visible(False)
    y_axis.spines["top"].set_visible(False)
    y_axis.spines["bottom"].set_visible(False)
    y_axis.set_xticks([])
    y_axis.set_yticks([])
    y_axis.set_ylabel("$z_2$", fontsize=36)

    save_image(fig, path)

def logging_progress(epoch: int, i: int, total_batches: int, progress_length: int=20, print_str=''):
    current = i + 1
    percent = round(current / total_batches * 100)
    curr_bar = int(current / total_batches * progress_length)

    bar = '*' * curr_bar + ' ' * (progress_length - curr_bar)
    end_char = '\n' if current == total_batches else '\r'

    print(f'Epoch {epoch}\t {percent:3d}% [{bar}] {print_str}', end=end_char, flush=True)