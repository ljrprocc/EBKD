import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tqdm
import os
import random

class DataSubSet(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds =inds

    def __getitem__(self, idx):
        base_ind = self.inds[idx]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)

def cycle(loader):
    while True:
        for data in loader:
            yield data

    
def get_dataloaders_and_datasets(base_dataset, test_base_dataset, opt, train_sampler=None):
    full_train = base_dataset
    all_inds = list(range(len(full_train)))
    # Set seed
    np.random.seed(1234)
    # Shuffle
    np.random.shuffle(all_inds)
    if opt.dataset == 'imagenet':
        save_dir = '/data/lijingru/imagenet_np/random'
    if opt.n_valid is not None:
        valid_inds, train_inds = all_inds[:opt.n_valid], all_inds[opt.n_valid:]
    else:
        valid_inds, train_inds = [], all_inds
    
    train_inds = np.array(train_inds)
    # print('************8')
    train_labeled_inds = []
    other_inds = []
    # print(len(train_inds))
    # print(train_inds.shape)
    # print('Preprocessing data:')
    trains = []
    load_local=False
    if opt.dataset == 'imagenet':
        i = 0
        while True:
            if os.path.isfile(save_dir + '{}.npy'.format(i)):
                i += 1
            else:
                break
        if i > 0:
            train_idx = random.randint(0, i)
            train_labels = np.load(save_dir + '{}.npy'.format(train_idx))
        if not load_local or i == 0:
            for ind in tqdm.tqdm(train_inds, desc='Processing data'):
                trains.append(full_train[ind][1])
            train_labels = np.array(trains)
            np.save(save_dir + '{}.npy'.format(i), train_labels)

    else:
        train_labels = np.array([full_train[ind][1] for ind in train_inds])
    
    if opt.labels_per_class > 0:
        for i in range(opt.n_cls):
            train_labeled_inds.extend(train_inds[train_labels == i][:opt.labels_per_class])
            other_inds.extend(train_inds[train_labels == i][opt.labels_per_class:])
    else:
        train_labeled_inds = train_inds

    dset_train = DataSubSet(
        base_dataset,
        inds=train_inds
    )
    dset_train_labeled = DataSubSet(
        base_dataset,
        inds=train_labeled_inds
    )
    dset_valid = DataSubSet(
        base_dataset,
        inds=valid_inds
    )
    
    if train_sampler is None:
        dload_train = DataLoader(
            dset_train,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            drop_last=True
        )
        dload_train_labeled = DataLoader(
            dset_train_labeled,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            drop_last=True
        )
        dload_train_labeled = cycle(dload_train_labeled)
    else:
        dload_train = DataLoader(
            dset_train,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=train_sampler
        )
        dload_train_labeled = DataLoader(
            dset_train_labeled,
            batch_size=opt.batch_size,
            pin_memory=True,
            num_workers=opt.num_workers,
            sampler=train_sampler
        )
        dload_train_labeled = cycle(dload_train_labeled)

    
    dset_test = test_base_dataset
    dload_valid = DataLoader(
        dset_valid,
        batch_size=25,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    dload_test = DataLoader(
        dset_test,
        batch_size=25,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    return dload_train, dload_train_labeled, dload_valid, dload_test
    
