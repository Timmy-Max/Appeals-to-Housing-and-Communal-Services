"""This file tuns training for the network"""
import src.models.bert_classifier as bc

classifier = bc.BertClassifier(
    model_path='cointegrated/rubert-tiny',
    tokenizer_path='cointegrated/rubert-tiny',
    n_classes=15,
    epochs=6
)
path_to_data = r'D:\Projects\Appeals-to-Housing-and-Communal-services\data\processed\data.csv'
classifier.preparation(path_to_data, lr=2e-5, batch_size=32)
classifier.train()
