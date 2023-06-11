"""This file shows metrics on test data"""
import seaborn as sns
import matplotlib.pyplot as plt
from src.models.compute_metrics import compute_metrics

_1, _2, _3, _4, _5, _6 = compute_metrics(
    r"models/bert_weighted.pt", path_to_data="data/processed/data.csv"
)
accuracy, balanced_accuracy = _1, _2
precision, recall, f1score = _3, _4, _5
confusion_matrix = _6

sns.heatmap(confusion_matrix, annot=True, fmt="g")
plt.ylabel("Prediction", fontsize=13)
plt.xlabel("Actual", fontsize=13)
plt.title("Confusion Matrix", fontsize=17)
plt.show()
