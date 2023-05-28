"""The module implements functions to preprocess data."""
from typing import Tuple, Any

import pandas as pd
import pickle
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
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
    # my_stop_words = ['санкт', 'петербург', 'фотофиксация', 'прилагаться', 'номер', 'спасибо', 'пожалуйста',
    #                 'здравствуйте',
    #                 'добрый', 'день', 'ул', 'рф', 'россия', 'правительство', 'управлять', 'компания', 'суворовский',
    #                 'литовский', 'далее', 'закон', 'тсж', 'председатель', 'этаж', 'здравствуйте', 'ленинградские',
    #                 'отказывается', 'принимать', 'заявку', 'на', 'устранение', 'более', 'полугода', 'предпортовый',
    #                 'пр', 'д', 'пр', 'ветеранов', 'г', "шаумяна"]

    # districts_names = ['адмиралтейский', 'василеостровский', 'выборгский', 'калининский', 'кировский', 'колпинский',
    #                   'красногвардейский', 'красносельский', 'кронштадтский', 'курортный', 'московский', 'невский',
    #                   'петроградский', 'петродворцовый', 'приморский', 'пушкинский', 'фрунзенский', 'центральный']

    sentence = sentence.lower()  # to lower
    sentence = re.sub(emoji_pattern, '', sentence)  # delete emoji
    sentence = re.sub(r'\d\w{1,2}', '', sentence)  # delete digits
    sentence = re.sub(r'\d', '', sentence)  # delete digits
    sentence = re.sub(r'[^\w\s]', '', sentence)  # delete symbols
    sentence = re.sub(r'[a-zA-Z\s]+', ' ', sentence)  # delete english symbols
    # sentence = word_tokenize(sentence, language='russian')  # tokenize
    # sentence = [token for token in sentence if token not in ['й', 'м', 'ый', 'ая', 'ой', 'ий', 'ым', 'я']]
    # sentence = [token for token in sentence if token not in my_stop_words]
    # sentence = [token for token in sentence if token not in districts_names]
    # print(" ".join(sentence))
    return sentence


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


def one_hot(k: int, n_classes: int = 15) -> list:
    """Accepts class number, returns one-hot vector.

        Args:
            n_classes:
            k (int): class number

        Returns:
            list: one-hot vector
    """
    hot = [0] * n_classes
    hot[k] = 1.0
    return hot


def split(data_path: str) -> tuple[list, list, list, list]:
    """Accept path to dataset, returns train and test data (80, 20 %).

        Args:
            data_path (str): path to dataset

        Returns:
            data, labels (tuple[list, list, list, list]): train, validation, test data and labels
    """
    data = pd.read_csv(data_path, encoding='utf-8-sig')
    texts = list(data.loc[:, 'appeal_text'])
    labels = list(data.loc[:, 'category_name'])
    train_data, test_data, train_labels, test_labels = train_test_split(texts, labels, test_size=0.20)
    # train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=15 / 95)
    # train_labels = [one_hot(k) for k in train_labels]
    # val_labels = [one_hot(k) for k in val_labels]
    # test_data = [one_hot(k) for k in test_labels]
    # data = train_data, val_data, test_data
    # labels = train_labels, val_labels, test_labels
    return train_data, test_data, train_labels, test_labels


