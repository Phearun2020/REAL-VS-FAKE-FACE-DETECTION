import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(train=True):
    """
    Get transforms for the dataset.
    For training: includes stronger augmentation to improve generalization.
    For validation/test: no augmentation.
    """
    if train:
        transform_list = [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(degrees=10),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
        ]
    else:
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

    return transforms.Compose(transform_list)

def get_dataset(root_dir, train=True):
    """
    Create dataset using ImageFolder.
    Assumes root_dir has subdirectories for each class (e.g., 0/, 1/).
    Label mapping: 0=real, 1=fake
    """
    transform = get_transforms(train=train)
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    return dataset

def get_dataloader(root_dir, batch_size=32, shuffle=True, train=True, num_workers=4):
    """
    Create DataLoader for the dataset.
    """
    dataset = get_dataset(root_dir, train=train)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader

# Convenience functions for train, val, test
def get_train_dataloader(data_dir='data/dataset/train', batch_size=32, num_workers=4):
    return get_dataloader(data_dir, batch_size=batch_size, shuffle=True, train=True, num_workers=num_workers)

def get_val_dataloader(data_dir='data/dataset/validate', batch_size=32, num_workers=4):
    return get_dataloader(data_dir, batch_size=batch_size, shuffle=False, train=False, num_workers=num_workers)

def get_test_dataloader(data_dir='data/dataset/test', batch_size=32, num_workers=4):
    return get_dataloader(data_dir, batch_size=batch_size, shuffle=False, train=False, num_workers=num_workers)
