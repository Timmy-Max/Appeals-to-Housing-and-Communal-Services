import src.models.bert_classifier as bc


classifier = bc.BertClassifier(
        model_path='cointegrated/rubert-tiny',
        tokenizer_path='cointegrated/rubert-tiny',
        n_classes=15,
        epochs=1
    )
classifier.preparation('data/processed/data.csv')
classifier.train()