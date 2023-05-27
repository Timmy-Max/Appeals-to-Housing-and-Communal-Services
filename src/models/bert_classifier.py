"""The module implements BERT Classifier class"""
from typing import Any

import numpy as np
import torch
from numpy import ndarray

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from src.data.make_dataloaders import get_dataloaders
from src.features.preprocess import split


class BertClassifier:

    def __init__(self, model_path, tokenizer_path, n_classes=2, epochs=1, dropout=0.3, model_save_path='models/bert.pt'):
        """Init BERT Classifier class properties.

            Args:
            model_path (str): model name
            tokenizer_path (str): tokenizer name
            n_classes (int): how many classes to predict
            epochs (int): number of epochs
            model_save_path (str): path to save model
        """
        self.loss_fn = None
        self.scheduler = None
        self.optimizer = None
        self.train_data, self.val_data, self.test_data = None, None, None
        self.train_labels, self.val_labels, self.test_labels = None, None, None
        self.train_loader, self.val_loader, self.test_loader = None, None, None
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path
        self.max_len = 512
        self.epochs = epochs
        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.out_features, self.out_features // 2),
            torch.nn.BatchNorm1d(self.out_features // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.out_features // 2, self.out_features // 4),
            torch.nn.BatchNorm1d(self.out_features // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.out_features // 4, n_classes)
        )
        self.model.to(self.device)

    def preparation(self, path_to_data: str):
        """Preparation of data, optimizer, scheduler and loss.

            Args:
                path_to_data (str): path to processed data
        """
        _, __ = split(path_to_data)
        self.train_data, self.val_data, self.test_data = _
        self.train_labels, self.val_labels, self.test_labels = __
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(self.train_data,
                                                                               self.val_data,
                                                                               self.test_data,
                                                                               self.train_labels,
                                                                               self.val_labels,
                                                                               self.test_labels,
                                                                               self.tokenizer)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)
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
        count = 0
        for i, data in enumerate(self.train_loader):
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
            self.optimizer.zero_grad()
            count = i

        self.scheduler.step()
        train_acc = correct_predictions.double() / count
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
        count = 0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
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
                count = i

        val_acc = correct_predictions.double() / count
        val_loss = np.mean(losses)
        return val_acc, val_loss

    def train(self):
        """Full train of network."""
        best_accuracy = 0
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_acc, train_loss = self.fit()
            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval()
            print(f'Val loss {val_loss} accuracy {val_acc}')
            print('-' * 10)

            if val_acc > best_accuracy:
                torch.save(self.model, self.model_save_path)
                best_accuracy = val_acc

        self.model = torch.load(self.model_save_path)

    def predict(self, text) -> ndarray:
        """Prediction of the text.

            Args:
                text (str): input text

            Returns:
                (ndarray): predictions array
        """
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

        return prediction


if __name__ == '__main__':
    classifier = BertClassifier(
        model_path='cointegrated/rubert-tiny',
        tokenizer_path='cointegrated/rubert-tiny',
        n_classes=15,
        epochs=5
    )
    classifier.preparation('data/processed/data.csv')
    classifier.train()
