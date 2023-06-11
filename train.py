"""This file runs training of the network."""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.models.compute_metrics import compute_metrics
from src.models.bert_classifier import BertClassifier

n_classes = 15
weights = np.ones(n_classes) / n_classes
weights[[0, 11]] = weights[[0, 11]] / 5
weights = weights / weights.sum()

params = {
    "model_path": "cointegrated/rubert-tiny",
    "tokenizer_path": "cointegrated/rubert-tiny",
    "n_classes": n_classes,
    "epochs": 15,
    "dropout": 0.5,
    "max_len": 512,
    "model_save_path": r"models/bert_weighted.pt",
    "path_to_data": "data/processed/data.csv",
    "lr": 2e-5,
    "batch_size": 32,
    "weights": weights,
}

model = BertClassifier(**params)
model.train()

_1, _2, _3, _4, _5, _6 = compute_metrics(
    model, n_classes=n_classes, path_to_data="data/processed/data.csv"
)
accuracy, balanced_accuracy = _1, _2
precision, recall, f1score = _3, _4, _5
confusion_matrix = _6

sns.heatmap(confusion_matrix, annot=True, fmt="g")
plt.ylabel("Prediction", fontsize=13)
plt.xlabel("Actual", fontsize=13)
plt.title("Confusion Matrix", fontsize=17)
plt.show()
