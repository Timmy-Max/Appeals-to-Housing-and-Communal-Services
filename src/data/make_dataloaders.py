"""The module implements functions to make dataset for transformer format."""
import torch

from torch import Tensor
from torch.utils.data import Dataset
from typing import Any


class CustomDataset(Dataset):
    """The class implements functionality for creating your own dataset of the desired format."""

    def __init__(self, texts: list, targets: list, tokenizer: Any, max_len: int = 512):
        """The function init dataset

        Args:
            texts (list): list of texts
            targets (list): list of labels
            tokenizer (AutoTokenizer): tokenizer
            max_len (int = 512): maximum text length
        """
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """The function returns number of texts in dataset

        Returns:
            (int): number of texts in dataset
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, str | Tensor]:
        """The function returns number of texts in dataset

        Args:
            idx (int): index of item

        Returns:
            (dict[str, str | Tensor]): tokenized text, tokens indexes,
            attention mask, targets tensor
        """
        text = str(self.texts[idx])
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.int64),
        }
