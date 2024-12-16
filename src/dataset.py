import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(
                self.dataset[index]["input_ids"], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                self.dataset[index]["attention_mask"], dtype=torch.long
            ),
            "token_type_ids": torch.tensor(
                self.dataset[index]["token_type_ids"], dtype=torch.long
            ),
            "labels": torch.tensor(
                self.dataset[index]["one_hot_labels"], dtype=torch.float
            ),
        }
