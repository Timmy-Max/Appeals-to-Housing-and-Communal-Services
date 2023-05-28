"""The module implements BERT Classifier class"""
import numpy as np
import torch
import time
import pickle

from typing import Any
from numpy import ndarray
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from src.data.make_dataloaders import CustomDataset
from src.data.make_dataloaders import get_dataloaders
from src.features.preprocess import split


class BertClassifier:

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 n_classes: int = 15,
                 epochs: int = 1,
                 dropout: float = 0.3,
                 max_len: int = 512,
                 model_save_path='models/bert.pt'):
        """Init BERT Classifier class properties.

            Args:
            model_path (str): model name
            tokenizer_path (str): tokenizer name
            n_classes (int): how many classes to predict
            epochs (int): number of epochs
            dropout (float): dropout of classificator
            max_len (int): maximum text length
            model_save_path (str): path to save model
        """
        self.train_loader = None
        self.valid_loader = None
        self.train_set = None
        self.valid_set = None
        self.loss_fn = None
        self.scheduler = None
        self.optimizer = None
        self.n_classes = n_classes

        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path
        self.max_len = max_len
        self.epochs = epochs
        self.dropout = dropout
        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.out_features, self.out_features // 2),
            torch.nn.BatchNorm1d(self.out_features // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.out_features // 2, self.out_features // 4),
            torch.nn.BatchNorm1d(self.out_features // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.out_features // 4, self.n_classes)
        )
        self.model.to(self.device)

    def preparation(self, path_to_data: str, lr: float = 1e-5, batch_size: int = 32):
        """Preparation of data, optimizer, scheduler and loss.

            Args:
                path_to_data (str): path to processed data
                lr (float): learning rate
                batch_size (int): batch size
        """
        data_train, data_valid, y_train, y_valid = split(path_to_data)
        self.train_set = CustomDataset(data_train, y_train, self.tokenizer)
        self.valid_set = CustomDataset(data_valid, y_valid, self.tokenizer)

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=batch_size, shuffle=True)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def fit(self) -> tuple[float | Any, ndarray]:
        """One epoch training.

            Returns:
                (tuple[float | Any, ndarray]): train loss and accuracy
        """
        self.model = self.model.train()
        losses = []
        correct_predictions = 0

        for data in self.train_loader:
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(outputs.logits, dim=1)
            loss = self.loss_fn(outputs.logits, targets)

            correct_predictions += torch.sum(predictions == targets)

            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        train_acc = correct_predictions.double() / len(self.train_set)
        train_loss = np.mean(losses)
        return train_acc, train_loss

    def eval(self) -> tuple[float | Any, ndarray]:
        """Eval scores after one epoch training.

            Returns:
                (tuple[float | Any, ndarray]): validation loss and accuracy
        """
        self.model = self.model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for data in self.valid_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                predictions = torch.argmax(outputs.logits, dim=1)
                loss = self.loss_fn(outputs.logits, targets)
                correct_predictions += torch.sum(predictions == targets)
                losses.append(loss.item())

        val_acc = correct_predictions.double() / len(self.valid_set)
        val_loss = np.mean(losses)
        return val_acc, val_loss

    def train(self):
        """Full train of network."""
        best_accuracy = 0
        start_time = time.time()
        for epoch in range(self.epochs):
            start_epoch = time.time()
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_acc, train_loss = self.fit()
            print(f'Train loss {train_loss:.3f} accuracy {train_acc:.3f}')

            val_acc, val_loss = self.eval()
            print(f'Val loss {val_loss:.3f} accuracy {val_acc:.3f}')
            print(f'Train epoch time: {((time.time() - start_epoch) / 60):.2f} m')
            print('-' * 10)

            if val_acc > best_accuracy:
                torch.save(self.model, self.model_save_path)
                best_accuracy = val_acc

        print(f'Total train time {((time.time() - start_time) / 60):.2f} m')
        self.model = torch.load(self.model_save_path)

    def predict(self, text: str, label_encoder: Any) -> ndarray:
        """Prediction of the text.

            Args:
                text (str): input text
                label_encoder (Any): label encoder to return category name

            Returns:
                (ndarray): predictions array
        """
        self.model = self.model.eval()
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        out = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )

        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
        category_name = label_encoder.inverse_transfrom(prediction) + f": {prediction}"

        return category_name


if __name__ == '__main__':
    classifier = BertClassifier(
        model_path='cointegrated/rubert-tiny',
        tokenizer_path='cointegrated/rubert-tiny',
        n_classes=15,
        epochs=5
    )
    classifier.preparation('data/processed/data.csv')
    classifier.train()
