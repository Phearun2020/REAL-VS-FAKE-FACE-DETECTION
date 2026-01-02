import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class RemapLabelsDataset(Dataset):
    """
    Dataset wrapper to remap labels.
    Assumes original: fake=0, real=1
    Maps to: real=0, fake=1
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Remap: if label == 0 (fake), set to 1; if label == 1 (real), set to 0
        remapped_label = 1 - label  # 0 -> 1, 1 -> 0
        return image, remapped_label

def get_transforms(train=True):
    """
    Get transforms for the dataset.
    For training: includes random horizontal flip.
    For validation/test: no augmentation.
    """
    transform_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ]

    if train:
        transform_list.insert(1, transforms.RandomHorizontalFlip())

    return transforms.Compose(transform_list)

def get_dataset(root_dir, train=True):
    """
    Create dataset using ImageFolder.
    Assumes root_dir has subdirectories for each class (e.g., fake/, real/).
    Remaps labels to: 0=real, 1=fake
    """
    transform = get_transforms(train=train)
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    # Remap labels
    dataset = RemapLabelsDataset(dataset)
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
def get_train_dataloader(data_dir='data/classification/train', batch_size=32, num_workers=4):
    return get_dataloader(data_dir, batch_size=batch_size, shuffle=True, train=True, num_workers=num_workers)

def get_val_dataloader(data_dir='data/classification/val', batch_size=32, num_workers=4):
    return get_dataloader(data_dir, batch_size=batch_size, shuffle=False, train=False, num_workers=num_workers)

def get_test_dataloader(data_dir='data/classification/test', batch_size=32, num_workers=4):
    return get_dataloader(data_dir, batch_size=batch_size, shuffle=False, train=False, num_workers=num_workers)
