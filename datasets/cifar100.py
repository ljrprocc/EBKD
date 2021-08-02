from __future__ import print_function

import os
import socket
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from .datasubset import get_dataloaders_and_datasets

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
    else:
        data_folder = '/data/lijingru/img_sample_eval_10000/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    test_folder = '/data/lijingru/cifar10/'

    return data_folder, test_folder

class CIFAR100Gen(Dataset):
    def __init__(self, root, transform=None, target_transform=None, return_target=False):
        self.root = root
        self.files = os.listdir(root)
        self.transform = transform
        self.target_transform = target_transform
        self.return_target = return_target
    
    def __getitem__(self, idx):
        f = os.path.join(self.root, self.files[idx])
        target = int(self.files[idx].split('_')[2])

        img = Image.open(f)
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.return_target:
            return img, target, idx
        else:
            return img
    
    def __len__(self):
        return len(self.files)

class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR100Instance Dataset.
    """
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class CIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset.
    """
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_cifar100_dataloaders(opt, batch_size=128, num_workers=8, is_instance=False, use_subdataset=False):
    """
    cifar 100
    """
    data_folder, test_folder = get_data_folder()
    train_list = [
        transforms.Pad(4, padding_mode="reflect"),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    test_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    if use_subdataset:
        train_list += [lambda x: x + opt.data_noise * torch.randn_like(x)]
        test_list += [lambda x: x + opt.data_noise * torch.randn_like(x)]

    train_transform = transforms.Compose(train_list)
    test_transform = transforms.Compose(test_list)
    if not opt.datafree:
        if is_instance:
            if opt.dataset == 'cifar100':
                train_set = CIFAR100Instance(root=test_folder,
                                            download=True,
                                            train=True,
                                            transform=train_transform)
            else:
                train_set = CIFAR10Instance(root=test_folder,
                                            download=True,
                                            train=True,
                                            transform=train_transform)
            n_data = len(train_set)
        else:
            if opt.dataset == 'cifar100':
                train_set = datasets.CIFAR100(root=test_folder,
                                            download=True,
                                            train=True,
                                            transform=train_transform)
            else:
                train_set = datasets.CIFAR10(root=test_folder,
                                            download=True,
                                            train=True,
                                            transform=train_transform)
    else:
        train_set = CIFAR100Gen(root=data_folder, transform=train_transform, return_target=True)
        if is_instance:
            n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    if opt.dataset == 'cifar100':
        test_set = datasets.CIFAR100(root=test_folder,
                                    download=True,
                                    train=False,
                                    transform=test_transform)
    else:
        test_set = datasets.CIFAR10(root=test_folder,
                                    download=True,
                                    train=False,
                                    transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))
    if use_subdataset:
        dload_train, dload_train_labeled, dload_valid, dload_test = get_dataloaders_and_datasets(train_set, test_set, opt)
        train_loader = (dload_train, dload_train_labeled)
        test_loader = (dload_valid, dload_test)
    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader


class CIFAR100InstanceSample(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        if self.train:
            num_samples = len(self.train_data)
            label = self.train_labels
        else:
            num_samples = len(self.test_data)
            label = self.test_labels

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


def get_cifar100_dataloaders_sample(opt, batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0, use_subdataset=False):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_list = [
        transforms.Pad(4, padding_mode="reflect"),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    test_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    if use_subdataset:
        train_list += [lambda x: x + 0.03 * torch.randn_like(x)]
        test_list += [lambda x: x + 0.03 * torch.randn_like(x)]

    train_transform = transforms.Compose(train_list)
    test_transform = transforms.Compose(test_list)

    train_set = CIFAR100InstanceSample(root=data_folder,
                                       download=True,
                                       train=True,
                                       transform=train_transform,
                                       k=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent)
    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))
    if use_subdataset:
        dload_train, dload_train_labeled, dload_valid, dload_test = get_dataloaders_and_datasets(train_set, test_set, opt)
        train_loader = (dload_train, dload_train_labeled)
        test_loader = (dload_valid, dload_test)

    return train_loader, test_loader, n_data
