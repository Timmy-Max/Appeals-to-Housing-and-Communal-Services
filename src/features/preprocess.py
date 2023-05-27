"""The module implements functions to preprocess data."""
from typing import Tuple, Any

import pandas as pd
import pickle
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize


def encode_labels(data_path: str):
    """Accept path to dataset, saves dataset with encoded categories names.

        Args:
            data_path (str): path to dataset
    """
    data = pd.read_csv(data_path, encoding='utf-8-sig')
    label_encoder = LabelEncoder()
    data.loc[:, 'category_name'] = label_encoder.fit_transform(data.loc[:, 'category_name'])
    output_path = 'data/interim/encoded_labels.csv'
    data.to_csv(output_path, index=False, header=True, encoding='utf-8-sig')
    print(f"Dataset with encoded labels saved to path: {output_path}")
    output_path = 'src/features/label_encoder.pickle'
    with open(output_path, 'wb') as file:
        pickle.dump(label_encoder, file)
    print(f"Labels encoder saved to path: {output_path}")


def clear_sentence(sentence: str) -> str:
    """Accept sentence, returns a clear sentence.

        Args:
            sentence (str): input sentence

        Returns:
            str: cleared sentence
    """
    pattern = r'[^\w]'
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", re.UNICODE)
    sentence = re.sub(emoji_pattern, ' ', sentence)  # delete emoji
    sentence = re.sub(pattern, ' ', sentence)  # delete tech symbols
    sentence = sentence.lower()  # to lower
    tokens = word_tokenize(sentence, language='russian')  # tokenize
    tokens = [token for token in tokens if token not in ["й", "м", "ая", "ый", "ой", "ым"]]

    return " ".join(tokens)


def clear_data(data_path: str):
    """Accept path to dataset, saves dataset with encoded categories names.

        Args:
            data_path (str): path to dataset
    """
    data = pd.read_csv(data_path, encoding='utf-8-sig')
    data.loc[:, "appeal_text"] = data.loc[:, "appeal_text"].apply(clear_sentence)
    output_path = 'data/processed/data.csv'
    data.to_csv(output_path, index=False, header=True, encoding='utf-8-sig')
    print(f"Cleared dataset saved to path: {output_path}")


def split(data_path: str) -> tuple[tuple[list, list, list], tuple[list, list, list]]:
    """Accept path to dataset, returns train, validation and test data (80, 15, 5 %).

        Args:
            data_path (str): path to dataset

        Returns: data, labels (tuple[tuple[list, list, list], tuple[list, list, list]]): train, validation and test
        data and labels
    """
    data = pd.read_csv(data_path, encoding='utf-8-sig')
    texts = list(data.loc[:, 'appeal_text'])
    labels = list(data.loc[:, 'category_name'])
    train_data, test_data, train_labels, test_labels = train_test_split(texts, labels, test_size=0.05)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=15 / 95)
    data = train_data, val_data, test_data
    labels = train_labels, val_labels, test_labels
    return data, labels


if __name__ == '__main__':
    encode_labels('data/interim/extracted.csv')
    clear_data('data/interim/encoded_labels.csv')

