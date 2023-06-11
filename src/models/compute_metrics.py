"""This file is intended for calculating metrics."""
import torch
from numpy import ndarray

from src.models.bert_classifier import load_model, BertClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix


def compute_metrics(
    classifier: str | BertClassifier,
    n_classes: int = 15,
    path_to_data: str = "data/processed/data.csv",
) -> tuple[float, float, float, float, float, ndarray]:
    """Compute and returns metrics.

    Args:
        classifier (str|BertClassifier): path to saved model or model
        path_to_data (str): path to data
        n_classes (int): number of output classes

    Returns: (tuple[float, float, float, float, float, ndarray]): accuracy, balanced_accuracy, precision,
    recall, F1-score, confusion matrix
    """
    if isinstance(classifier, str):
        classifier = load_model(
            path_to_model=classifier,
            model_path="cointegrated/rubert-tiny",
            tokenizer_path="cointegrated/rubert-tiny",
            path_to_data=path_to_data,
            n_classes=n_classes,
        )

    predictions = []
    test_labels = []
    with torch.no_grad():
        for data in classifier.valid_loader:
            input_ids = data["input_ids"].to(classifier.device)
            attention_mask = data["attention_mask"].to(classifier.device)
            targets = data["targets"].to(classifier.device)

            outputs = classifier.model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            predictions.extend(
                torch.argmax(outputs.logits, dim=1).detach().cpu().numpy()
            )
            test_labels.extend(list(targets.detach().cpu().numpy()))

    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1score = precision_recall_fscore_support(
        test_labels, predictions, average="macro"
    )[:3]
    balanced_accuracy = balanced_accuracy_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions)
    print(f"Accuracy: {accuracy:.3f}, Balanced Accuracy: {balanced_accuracy:.3f}")
    print("The following metrics are obtained by macro-averaging")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1score:.3f}")
    return accuracy, balanced_accuracy, precision, recall, f1score, cm
