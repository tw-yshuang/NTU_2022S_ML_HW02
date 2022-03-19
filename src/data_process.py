import os, sys, gc
import torch
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.append(os.path.abspath(__package__))
from data_preprocess import preprocess_data


class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


def get_dataloader(data_dir: str = './Data/libriphone', mode: str = 'train', batch_size=32, n_workers=1):
    concat_nframes = 1

    """Generate dataloader"""
    data, labels = preprocess_data(
        split='train', feat_dir=f'{data_dir}/feat', phone_path=data_dir, concat_nframes=concat_nframes, train_ratio=1
    )

    dataset = LibriDataset(data, labels)
    del data, labels
    gc.collect()

    if mode == 'full':
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=False,
            # collate_fn=collate_batch,
        )
        return loader, 39 * concat_nframes

    else:
        # Split dataset into training dataset and validation dataset
        trainlen = int(0.8 * len(dataset))
        lengths = [trainlen, len(dataset) - trainlen]
        trainset, validset = random_split(dataset, lengths)

        train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=False,
        )
        valid_loader = DataLoader(
            validset,
            batch_size=batch_size,
            num_workers=n_workers,
            drop_last=False,
            pin_memory=False,
            # collate_fn=collate_batch,
        )

        return train_loader, valid_loader, 39 * concat_nframes


if __name__ == '__main__':
    train_loader, valid_loader, speaker_num = get_dataloader('./Data/libriphone', batch_size=128)
    print(f"train_loader: {len(train_loader)} dataset")
    print(f"valid_loader: {len(valid_loader)} dataset")
    print(f"speaker_num: {speaker_num}")
