import matplotlib.pyplot as plt
from typing import Any


def save_images(images: Any, row: int, col: int, path: str, epoch: int):
    fig = plt.figure()
    for i in range(row * col):
        ax = fig.add_subplot(row, col, i+1)
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.suptitle(f"Epoch {epoch}")
    fig.savefig(path)
    plt.close(fig)


def logging_progress(epoch: int, i: int, total_batches: int, progress_length: int=20, print_str=''):
    current = i + 1
    percent = round(current / total_batches * 100)
    curr_bar = int(current / total_batches * progress_length)

    bar = '*' * curr_bar + ' ' * (progress_length - curr_bar)
    end_char = '\n' if current == total_batches else '\r'

    print(f'Epoch {epoch}\t {percent:3d}% [{bar}] {print_str}', end=end_char, flush=True)