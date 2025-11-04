import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math



def save_img(img: np.array, dst):

    assert Path(dst).suffix != "" , 'define file type'

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    img = Image.fromarray(img)
    img.save(dst)


def save_couple(img1: tuple[np.array, str], img2:tuple[np.array, str], dst):

    img1, name1 = img1
    img2, name2 = img2

    dst = Path(dst)

    assert dst.suffix != "", 'define file type'

    os.makedirs(dst.parent, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(19.2, 10.8))

    axs[0].imshow(img1)
    axs[0].set_title('name1')
    axs[0].axis('off')
    axs[1].imshow(img2)
    axs[1].set_title('name2')
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(dst, dpi=100)
    plt.close()


def plot_multiple(images: list[tuple[np.array, str]], save_dst=None, plot=True):

    num_imgs = len(images)

    # Calculate the number of columns and rows
    cols = math.floor(math.sqrt(num_imgs) * 1.5)  # Increase columns for rectangle
    rows = math.ceil(num_imgs / cols)  # Rows as a function of columns

    f, axs = plt.subplots(rows, cols, figsize=(19.2, 10.8))

    axs = axs.flatten()

    for i, (img, name) in enumerate(images):
        axs[i].imshow(img)
        axs[i].set_title(name)
        axs[i].axis('off')

    plt.tight_layout()

    if save_dst:
        assert save_dst.suffix != "", 'define file type'
        os.makedirs(save_dst.parent, exist_ok=True)

        plt.savefig(save_dst, dpi=100)

    if plot:
        plt.show()

    plt.close()
