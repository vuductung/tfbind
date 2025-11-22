import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.processing import dna_to_one_hot_encode


def load_dataset(path):
    loaded_ds = pd.read_csv(path)
    return {
        "labels": loaded_ds["label"].values,
        "sequences": [dna_to_one_hot_encode(sequence).T for sequence in loaded_ds["sequence"]],
    }


class DnaDataset(Dataset):
    def __init__(self, labels, sequences):
        self.labels = labels
        self.sequences = sequences

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)

        return {"seq": sequence, "label": label}

    def __len__(self):
        return len(self.sequences)


def collate_fn(batch):
    sequences = [item["seq"] for item in batch]
    labels = [item["label"] for item in batch]
    lengths = torch.tensor([len(item["seq"]) for item in batch])
    labels = torch.hstack(labels).unsqueeze(1)

    sequences = pad_sequence(sequences, batch_first=True)

    return {"seq": sequences, "labels": labels, "lengths": lengths}
