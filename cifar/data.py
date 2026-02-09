import os
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset
from robustbench.data import _load_dataset, CORRUPTIONS, PREPROCESSINGS

DEFAULT_DATA_DIR = os.path.expanduser("~")


def load_svhn(
    n_examples: Optional[int] = None,
    data_dir: str = DEFAULT_DATA_DIR,
    transforms_test: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.SVHN(root=data_dir,
                            split='test',
                            transform=transforms_test,
                            download=False)
    return _load_dataset(dataset, n_examples)


def load_svhn_c(
    n_examples: int,
    severity: int = 5,
    data_dir: str = DEFAULT_DATA_DIR,
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    _: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert 1 <= severity <= 5
    n_total_svhn = 26032

    data_dir = Path(data_dir)
    data_root_dir = data_dir / 'SVHN-C'

    labels_path = data_root_dir / 'labels.npy'
    labels = np.load(labels_path)

    x_test_list, y_test_list = [], []
    n_pert = len(corruptions)
    for corruption in corruptions:
        corruption_file_path = data_root_dir / (corruption + '.npy')

        images_all = np.load(corruption_file_path)
        images = images_all[(severity - 1) * n_total_svhn:severity *
                            n_total_svhn]
        n_img = int(np.ceil(n_examples / n_pert))
        x_test_list.append(images[:n_img])
        y_test_list.append(labels[:n_img])

    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
    if shuffle:
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]

    x_test = np.transpose(x_test, (0, 3, 1, 2))
    x_test = x_test.astype(np.float32) / 255
    x_test = torch.tensor(x_test)[:n_examples]
    y_test = torch.tensor(y_test)[:n_examples]

    return x_test, y_test

def load_tiny_imagenet_c(
    n_examples: Optional[int] = 10000,
    severity: int = 5,
    data_dir: str = DEFAULT_DATA_DIR,
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    prepr: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 10000:
        raise ValueError(
            'The evaluation is currently possible on at most 10000 points.')

    assert len(
        corruptions
    ) == 1, "so far only one corruption is supported (that's how this function is called in eval.py"

    data_folder_path = Path(data_dir) / 'Tiny-ImageNet-C' / corruptions[0] / str(severity)
    tiny_imagenet_c = ImageFolder(data_folder_path, transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))

    repeats = n_examples // len(tiny_imagenet_c) + 1
    repeated_tiny_imagenet_c = data.ConcatDataset([tiny_imagenet_c] * repeats)
    test_loader = data.DataLoader(repeated_tiny_imagenet_c,
                                  batch_size=n_examples,
                                  shuffle=shuffle,
                                  num_workers=2)

    x_test, y_test = next(iter(test_loader))

    return x_test, y_test

def load_places365_c(
    n_examples: Optional[int] = 10000,
    severity: int = 5,
    data_dir: str = DEFAULT_DATA_DIR,
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    prepr: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 10000:
        raise ValueError(
            'The evaluation is currently possible on at most 10000 points.')

    assert len(
        corruptions
    ) == 1, "so far only one corruption is supported (that's how this function is called in eval.py"

    data_folder_path = Path(data_dir) / 'PLACES365-C' / corruptions[0] / str(severity)
    tiny_imagenet_c = ImageFolder(data_folder_path, transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))

    repeats = n_examples // len(tiny_imagenet_c) + 1
    repeated_tiny_imagenet_c = data.ConcatDataset([tiny_imagenet_c] * repeats)
    test_loader = data.DataLoader(repeated_tiny_imagenet_c,
                                  batch_size=n_examples,
                                  shuffle=shuffle,
                                  num_workers=2)

    x_test, y_test = next(iter(test_loader))

    return x_test, y_test

def load_textures_c(
    n_examples: Optional[int] = 10000,
    severity: int = 5,
    data_dir: str = DEFAULT_DATA_DIR,
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    prepr: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 10000:
        raise ValueError(
            'The evaluation is currently possible on at most 10000 points.')

    assert len(
        corruptions
    ) == 1, "so far only one corruption is supported (that's how this function is called in eval.py"

    data_folder_path = Path(data_dir) / 'Texture-C' / corruptions[0] / str(severity)
    tiny_imagenet_c = ImageFolder(data_folder_path, transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))

    repeats = n_examples // len(tiny_imagenet_c) + 1
    repeated_tiny_imagenet_c = data.ConcatDataset([tiny_imagenet_c] * repeats)
    test_loader = data.DataLoader(repeated_tiny_imagenet_c,
                                  batch_size=n_examples,
                                  shuffle=shuffle,
                                  num_workers=2)

    x_test, y_test = next(iter(test_loader))

    return x_test, y_test

from typing import Optional, Tuple, Callable, Sequence
import torch
import numpy as np
from numpy.random import RandomState
from pathlib import Path
import os
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torchvision import transforms


def load_gaussian(
    n_examples: Optional[int] = 10000,
    shape: Tuple[int, int, int] = (32, 32, 3),
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

    transformed_imgs = torch.stack([torch.tensor(img).permute(2, 0, 1).float() / 255.0 for img in imgs])

    dataset = TensorDataset(transformed_imgs, labels)

    repeats = n_examples // len(dataset) + 1
    repeated_dataset = ConcatDataset([dataset] * repeats)

    loader = DataLoader(
        repeated_dataset, batch_size=n_examples, shuffle=shuffle, num_workers=2
    )

    x_test, y_test = next(iter(loader))

    return x_test[:n_examples], y_test[:n_examples]

def load_uniform(
    n_examples: Optional[int] = 10000,
    shape: Tuple[int, int, int] = (32, 32, 3),
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

    transformed_imgs = torch.stack([torch.tensor(img).permute(2, 0, 1).float() / 255.0 for img in imgs])

    dataset = TensorDataset(transformed_imgs, labels)

    repeats = n_examples // len(dataset) + 1
    repeated_dataset = ConcatDataset([dataset] * repeats)

    loader = DataLoader(
        repeated_dataset, batch_size=n_examples, shuffle=shuffle, num_workers=2
    )

    x_test, y_test = next(iter(loader))

    return x_test[:n_examples], y_test[:n_examples]
