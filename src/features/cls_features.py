"""Module implements function that creates dataset with CLS embedding for each sentence"""
import torch
import pandas as pd

from src.models.bert_classifier import load_model


def save_cls_features(
    path_to_model: str,
    path_to_save: str = "data/processed/cls_features/cls_features.py",
    model_path: str = "cointegrated/rubert-tiny",
    tokenizer_path: str = "cointegrated/rubert-tiny",
    path_to_data: str = "data/processed/data.csv",
    n_classes: int = 15,
):
    """Function creates dataset with CLS embedding for each sentence.

    Args:
        path_to_model (str): path to saved model
        path_to_save (str): path to save dataset
        model_path (str): model name
        tokenizer_path (str): tokenizer name
        path_to_data (str): path to data
        n_classes (int): number of classification classes
    """
    classifier = load_model(
        path_to_model=path_to_model,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        path_to_data=path_to_data,
        n_classes=n_classes,
    )
    embeddings_data = []
    with torch.no_grad():
        for data in classifier.train_loader:
            input_ids = data["input_ids"].to(classifier.device)
            attention_mask = data["attention_mask"].to(classifier.device)
            targets = list(data["targets"].cpu().numpy())
            text = data["text"]

            outputs = classifier.model.bert(
                input_ids=input_ids, attention_mask=attention_mask
            )
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings).cpu().numpy()
            embeddings = [list(embedding) for embedding in embeddings]
            embeddings_data.extend(zip(embeddings, text, targets))

    embeddings_data = pd.DataFrame(
        embeddings_data, columns=["embedding", "text", "category"]
    )
    embeddings_data.to_csv(path_to_save, index=False, header=True)
    print(f"Embeddings dataset saved to path: {path_to_save}")
