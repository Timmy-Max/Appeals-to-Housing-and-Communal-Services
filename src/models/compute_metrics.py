"""This file is intended for calculating metrics."""
import torch
import src.models.bert_classifier as bc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, balanced_accuracy_score

classifier = bc.BertClassifier(
    model_path='cointegrated/rubert-tiny',
    tokenizer_path='cointegrated/rubert-tiny',
    n_classes=15,
    epochs=6
)

path_to_model = r'D:\Projects\Appeals-to-Housing-and-Communal-services\models\bert.pt'
classifier.model = torch.load(path_to_model)

path_to_data = r'D:\Projects\Appeals-to-Housing-and-Communal-services\data\processed\data.csv'
classifier.preparation(path_to_data, lr=2e-5, batch_size=32)

predictions = []
test_labels = []
with torch.no_grad():
    for data in classifier.valid_loader:
        input_ids = data["input_ids"].to(classifier.device)
        attention_mask = data["attention_mask"].to(classifier.device)
        targets = data["targets"].to(classifier.device)

        outputs = classifier.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predictions.extend(torch.argmax(outputs.logits, dim=1).detach().cpu().numpy())
        test_labels.extend(list(targets.detach().cpu().numpy()))

accuracy = accuracy_score(test_labels, predictions)
precision, recall, f1score = precision_recall_fscore_support(test_labels, predictions, average='micro')[:3]
balanced_accuracy = balanced_accuracy_score(test_labels, predictions)
print(f'accuracy: {accuracy:.3f}, balanced accuracy: {balanced_accuracy:.3f}')
print(f'precision: {precision:.3f}, recall: {recall:.3f}, f1score: {f1score:.3f}')

