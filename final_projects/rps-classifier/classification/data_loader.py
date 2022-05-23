import os

from torch.utils.data import DataLoader, random_split, ConcatDataset

from torchvision import transforms
from torchvision.datasets import ImageFolder

def load_dataset(
    data_transforms,
    dataset_dir='./classification/dataset',
    dataset_name='rps',
    is_train=True,
):
    dataset_dir = os.path.join(dataset_dir, dataset_name)
    is_train_dir = 'train' if is_train else 'test'

    dataset = ImageFolder(
        os.path.join(dataset_dir, is_train_dir),
        data_transforms[is_train_dir]
    )
    return dataset


def get_loaders(config, input_size):
    data_transforms_augmentations = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'test': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'test': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
    }

    dataset_name = config.dataset_name
    train_set1 = load_dataset(
        data_transforms, dataset_name=dataset_name, is_train=True
    )
    test_set1 = load_dataset(
        data_transforms, dataset_name=dataset_name, is_train=False
    )

    train_set2 = load_dataset(
        data_transforms_augmentations, dataset_name=dataset_name, is_train=True
    )
    test_set2 = load_dataset(
        data_transforms_augmentations, dataset_name=dataset_name, is_train=False
    )

    train_set = ConcatDataset([train_set1, train_set2])
    test_set = ConcatDataset([test_set1, test_set2])

    # Shuffle dataset to split into valid/test set.
    train_set, valid_set, test_set = divide_dataset(
        train_set, test_set, config.dataset_name,
        config.train_ratio, config.valid_ratio, config.test_ratio
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader


def divide_dataset(
    train_set,
    test_set,
    data_name='rps',
    train_ratio=.6,
    valid_ratio=.2,
    test_ratio=.2
):
    if data_name == 'rps':
        train_cnt = int(len(train_set) * train_ratio)
        valid_cnt = int(len(train_set) * valid_ratio)
        test_cnt = len(train_set) - train_cnt - valid_cnt

        train_set, valid_set, test_set = random_split(
            train_set,
            [train_cnt, valid_cnt, test_cnt]
        )
    else:
        raise NotImplementedError('You need to specify dataset name.')

    return train_set, valid_set, test_set
