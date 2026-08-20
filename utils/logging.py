import numpy as np
import matplotlib.pyplot as plt
from typing import Any


def save_images(images: Any, row: int, col: int, path: str, epoch: int):
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
    fig = save_images(images, row, col, path, epoch)
    fig.suptitle(f"Epoch {epoch}")
    save_image(fig, path)

def save_gan_test_image(images: Any, row: int, col: int, path: str, epoch: int):
    fig = save_images(images, row, col, path, epoch)
    
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
    fig = save_images(images, row, col, path, epoch)


def logging_progress(epoch: int, i: int, total_batches: int, progress_length: int=20, print_str=''):
    current = i + 1
    percent = round(current / total_batches * 100)
    curr_bar = int(current / total_batches * progress_length)

    bar = '*' * curr_bar + ' ' * (progress_length - curr_bar)
    end_char = '\n' if current == total_batches else '\r'

    print(f'Epoch {epoch}\t {percent:3d}% [{bar}] {print_str}', end=end_char, flush=True)