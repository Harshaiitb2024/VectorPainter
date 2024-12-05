# -*- coding: utf-8 -*-
# Author: ximing
# Copyright (c) 2023, XiMing Xing.
# License: MPL-2.0 License
import os
from typing import AnyStr, Union, Text, List
import pathlib

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from .misc import AnyPath


def view_images(
        images: Union[np.ndarray, List[np.ndarray]],
        num_rows: int = 1,
        offset_ratio: float = 0.02,
        save_image: bool = False,
        fp: Union[Text, pathlib.Path, os.PathLike] = None,
) -> Image:
    if save_image:
        assert fp is not None

    if isinstance(images, list):
        images = np.concatenate(images, axis=0)

    if isinstance(images, np.ndarray) and images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images] if not isinstance(images, list) else images
        num_empty = len(images) % num_rows

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    # Calculate the composite image
    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = int(np.ceil(num_items / num_rows))  # count the number of columns
    image_h = h * num_rows + offset * (num_rows - 1)
    image_w = w * num_cols + offset * (num_cols - 1)
    assert image_h > 0, "Invalid image height: {} (num_rows={}, offset_ratio={}, num_items={})".format(
        image_h, num_rows, offset_ratio, num_items)
    assert image_w > 0, "Invalid image width: {} (num_cols={}, offset_ratio={}, num_items={})".format(
        image_w, num_cols, offset_ratio, num_items)
    image_ = np.ones((image_h, image_w, 3), dtype=np.uint8) * 255

    # Ensure that the last row is filled with empty images if necessary
    if len(images) % num_cols > 0:
        empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
        num_empty = num_cols - len(images) % num_cols
        images += [empty_images] * num_empty

    for i in range(num_rows):
        for j in range(num_cols):
            k = i * num_cols + j
            if k >= num_items:
                break
            image_[i * (h + offset): i * (h + offset) + h, j * (w + offset): j * (w + offset) + w] = images[k]

    pil_img = Image.fromarray(image_)
    if save_image:
        pil_img.save(fp)
    return pil_img


def save_image(image_array: np.ndarray, fname: AnyPath):
    image = np.transpose(image_array, (1, 2, 0)).astype(np.uint8)
    pil_image = Image.fromarray(image)
    pil_image.save(fname)


def plot_couple(input_1: torch.Tensor,
                input_2: torch.Tensor,
                step: int,
                output_dir: str,
                fname: AnyPath,  # file name
                prompt: str = '',  # text prompt as image tile
                pad_value: float = 0,
                dpi: int = 300):
    if input_1.shape != input_2.shape:
        raise ValueError("inputs and outputs must have the same dimensions")

    plt.figure()
    plt.subplot(1, 2, 1)  # nrows=1, ncols=2, index=1
    grid = make_grid(input_1, normalize=True, pad_value=pad_value)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title("Input")

    plt.subplot(1, 2, 2)  # nrows=1, ncols=2, index=2
    grid = make_grid(input_2, normalize=True, pad_value=pad_value)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title(f"Rendering - {step} steps")

    def insert_newline(string, point=9):
        # split by blank
        words = string.split()
        if len(words) <= point:
            return string

        word_chunks = [words[i:i + point] for i in range(0, len(words), point)]
        new_string = "\n".join(" ".join(chunk) for chunk in word_chunks)
        return new_string

    plt.suptitle(insert_newline(prompt), fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{fname}.png", dpi=dpi)
    plt.close()


def plot_img(inputs: torch.Tensor,
             output_dir: AnyStr,
             fname: AnyPath,  # file name
             pad_value: float = 0):
    assert torch.is_tensor(inputs), f"The input must be tensor type, but got {type(inputs)}"

    grid = make_grid(inputs, normalize=True, pad_value=pad_value)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    # plt.imshow(ndarr)
    # plt.axis("off")
    # plt.tight_layout()
    # plt.close()

    im = Image.fromarray(ndarr)
    im.save(f"{output_dir}/{fname}.png")


def plot_img_title(inputs: torch.Tensor,
                   title: str,
                   output_dir: AnyStr,
                   fname: AnyPath,  # file name
                   pad_value: float = 0,
                   dpi: int = 500):
    assert torch.is_tensor(inputs), f"The input must be tensor type, but got {type(inputs)}"

    grid = make_grid(inputs, normalize=True, pad_value=pad_value)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title(f"{title}")
    plt.savefig(f"{output_dir}/{fname}.png", dpi=dpi)
    plt.close()
