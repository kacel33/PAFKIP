from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple
import os

import numpy as np
import torch
import torch.utils.data as data

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset
from robustbench.data import _load_dataset, CORRUPTIONS, PREPROCESSINGS

from typing import Optional, Tuple, Callable, Sequence
import torch
import numpy as np
from numpy.random import RandomState
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torchvision import transforms

DEFAULT_DATA_DIR = os.path.expanduser("~/imagenet")


def load_imagenet_o_c(
    n_examples: Optional[int] = 5000,
    severity: int = 5,
    data_dir: str = DEFAULT_DATA_DIR,
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    prepr: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 5000:
        raise ValueError(
            'The evaluation is currently possible on at most 5000 points.')

    assert len(
        corruptions
    ) == 1, "so far only one corruption is supported (that's how this function is called in eval.py"
    # TODO: generalize this (although this would probably require writing a function similar to `load_corruptions_cifar`
    #  or alternatively creating yet another CustomImageFolder class that fetches images from multiple corruption types
    #  at once -- perhaps this is a cleaner solution)

    data_folder_path = Path(data_dir) / 'ImageNet-O-C' / corruptions[0] / str(severity)
    imagenet_o_c = ImageFolder(data_folder_path, prepr)
    repeats = n_examples // len(imagenet_o_c) + 1
    repeated_imagenet_o_c = data.ConcatDataset([imagenet_o_c] * repeats)
    test_loader = data.DataLoader(repeated_imagenet_o_c,
                                  batch_size=n_examples,
                                  shuffle=shuffle,
                                  num_workers=4)

    x_test, y_test = next(iter(test_loader))

    return x_test, y_test

def load_places_c(
    n_examples: Optional[int] = 5000,
    severity: int = 5,
    data_dir: str = DEFAULT_DATA_DIR,
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    prepr: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 5000:
        raise ValueError(
            'The evaluation is currently possible on at most 5000 points.')

    assert len(
        corruptions
    ) == 1, "so far only one corruption is supported (that's how this function is called in eval.py"
    # TODO: generalize this (although this would probably require writing a function similar to `load_corruptions_cifar`
    #  or alternatively creating yet another CustomImageFolder class that fetches images from multiple corruption types
    #  at once -- perhaps this is a cleaner solution)

    data_folder_path = Path(data_dir) / 'PLACES365-C' / corruptions[0] / str(severity)
    imagenet_o_c = ImageFolder(data_folder_path, prepr)
    repeats = n_examples // len(imagenet_o_c) + 1
    repeated_imagenet_o_c = data.ConcatDataset([imagenet_o_c] * repeats)
    test_loader = data.DataLoader(repeated_imagenet_o_c,
                                  batch_size=n_examples,
                                  shuffle=shuffle,
                                  num_workers=4)

    x_test, y_test = next(iter(test_loader))

    return x_test, y_test

def load_textures_c(
    n_examples: Optional[int] = 5000,
    severity: int = 5,
    data_dir: str = DEFAULT_DATA_DIR,
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    prepr: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 5000:
        raise ValueError(
            'The evaluation is currently possible on at most 5000 points.')

    assert len(
        corruptions
    ) == 1, "so far only one corruption is supported (that's how this function is called in eval.py"
    # TODO: generalize this (although this would probably require writing a function similar to `load_corruptions_cifar`
    #  or alternatively creating yet another CustomImageFolder class that fetches images from multiple corruption types
    #  at once -- perhaps this is a cleaner solution)

    data_folder_path = Path(data_dir) / 'Textures-C' / corruptions[0] / str(severity)
    imagenet_o_c = ImageFolder(data_folder_path, prepr)
    repeats = n_examples // len(imagenet_o_c) + 1
    repeated_imagenet_o_c = data.ConcatDataset([imagenet_o_c] * repeats)
    test_loader = data.DataLoader(repeated_imagenet_o_c,
                                  batch_size=n_examples,
                                  shuffle=shuffle,
                                  num_workers=4)

    x_test, y_test = next(iter(test_loader))

    return x_test, y_test

def load_gaussian(
    n_examples: Optional[int] = 5000,
    shape: Tuple[int, int, int] = (224, 224, 3),
    seed: int = 1,
    data_dir: str = DEFAULT_DATA_DIR,
    shuffle: bool = False,
    transform: Callable = transforms.Compose([
        transforms.ToTensor(),
    ]),
    ckpt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads a Gaussian noise dataset, similar to the structure of `load_textures_c`.

    Args:
        n_examples (int): Number of examples to generate.
        shape (tuple): Shape of each sample (default: (224, 224, 3)).
        seed (int): Seed for random number generation.
        data_dir (str): Directory for storing/checkpointing data.
        shuffle (bool): Whether to shuffle the dataset.
        transform (callable): Transformation to apply to the data.
        ckpt (str): Checkpoint directory to save/load generated data.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the dataset (x_test) and labels (y_test).
    """
    if n_examples > 10000:
        raise ValueError("The evaluation is currently possible on at most 10000 points.")

    # Setup checkpoint paths if provided
    if ckpt is not None:
        ckpt_path = Path(ckpt) / f"{shape[0]}_{n_examples}"
        img_path = ckpt_path / 'imgs.npy'
        ckpt_path.mkdir(parents=True, exist_ok=True)

        if not img_path.exists():
            rng = RandomState(seed)
            imgs = np.clip(rng.randn(n_examples, *shape) + 0.5, 0, 1) * 255
            imgs = imgs.astype(np.uint8)
            np.save(img_path, imgs)
        else:
            imgs = np.load(img_path)
    else:
        rng = RandomState(seed)
        imgs = np.clip(rng.randn(n_examples, *shape) + 0.5, 0, 1) * 255
        imgs = imgs.astype(np.uint8)

    labels = torch.tensor([-1] * n_examples, dtype=torch.long)

    # Apply transformations
    transformed_imgs = torch.stack([torch.tensor(img).permute(2, 0, 1).float() / 255.0 for img in imgs])

    # Create TensorDataset
    dataset = TensorDataset(transformed_imgs, labels)

    # Repeat dataset if n_examples exceeds dataset length
    repeats = n_examples // len(dataset) + 1
    repeated_dataset = ConcatDataset([dataset] * repeats)

    # Create DataLoader
    loader = DataLoader(
        repeated_dataset, batch_size=n_examples, shuffle=shuffle, num_workers=2
    )

    x_test, y_test = next(iter(loader))

    return x_test[:n_examples], y_test[:n_examples]

def load_uniform(
    n_examples: Optional[int] = 5000,
    shape: Tuple[int, int, int] = (224, 224, 3),
    seed: int = 1,
    data_dir: str = DEFAULT_DATA_DIR,
    shuffle: bool = False,
    transform: Callable = transforms.Compose([
        transforms.ToTensor(),
    ]),
    ckpt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads a Uniform noise dataset, similar to the structure of `load_textures_c`.

    Args:
        n_examples (int): Number of examples to generate.
        shape (tuple): Shape of each sample (default: (224, 224, 3)).
        seed (int): Seed for random number generation.
        data_dir (str): Directory for storing/checkpointing data.
        shuffle (bool): Whether to shuffle the dataset.
        transform (callable): Transformation to apply to the data.
        ckpt (str): Checkpoint directory to save/load generated data.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the dataset (x_test) and labels (y_test).
    """
    if n_examples > 10000:
        raise ValueError("The evaluation is currently possible on at most 10000 points.")

    # Setup checkpoint paths if provided
    if ckpt is not None:
        ckpt_path = Path(ckpt) / f"{shape[0]}_{n_examples}"
        img_path = ckpt_path / 'imgs.npy'
        ckpt_path.mkdir(parents=True, exist_ok=True)

        if not img_path.exists():
            rng = RandomState(seed)
            imgs = np.clip(rng.rand(n_examples, *shape), 0, 1) * 255
            imgs = imgs.astype(np.uint8)
            np.save(img_path, imgs)
        else:
            imgs = np.load(img_path)
    else:
        rng = RandomState(seed)
        imgs = np.clip(rng.rand(n_examples, *shape), 0, 1) * 255
        imgs = imgs.astype(np.uint8)

    labels = torch.tensor([-1] * n_examples, dtype=torch.long)

    # Apply transformations
    transformed_imgs = torch.stack([torch.tensor(img).permute(2, 0, 1).float() / 255.0 for img in imgs])

    # Create TensorDataset
    dataset = TensorDataset(transformed_imgs, labels)

    # Repeat dataset if n_examples exceeds dataset length
    repeats = n_examples // len(dataset) + 1
    repeated_dataset = ConcatDataset([dataset] * repeats)

    # Create DataLoader
    loader = DataLoader(
        repeated_dataset, batch_size=n_examples, shuffle=shuffle, num_workers=2
    )

    x_test, y_test = next(iter(loader))

    return x_test[:n_examples], y_test[:n_examples]

