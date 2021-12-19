import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def load_split_train_test(
    dataset_directory: str, test_size=0.2, batch_size=4
) -> tuple[DataLoader, DataLoader]:
    """
    Splitting dataset to train and test sets. Creating loaders for each set.

    Parameters
    ----------
    dataset_directory: string
        directory of image dataset
    test_size: float
        percentage size of test dataset
    batch_size: int
        size of batch for training

    Returns
    -------
    train_loader, test_loader: tuple[DataLoader, DataLoader]
        two DataLoaders of train and test set ready to be loaded into training loop

    """
    train_transforms = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor()]
    )

    train_data = datasets.ImageFolder(dataset_directory, transform=train_transforms)
    test_data = datasets.ImageFolder(dataset_directory, transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(test_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return train_loader, test_loader
