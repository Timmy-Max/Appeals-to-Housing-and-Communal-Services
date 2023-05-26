""""The module implements functions to preprocess data"""

import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder


def encode_labels(data_path: str):
    """Accept path to dataset, saves dataset with encoded categories names.

            Args:
                data_path (str): path to files
    """
    data = pd.read_csv(data_path, encoding='utf-8-sig')
    label_encoder = LabelEncoder()
    data.loc['category_name'] = label_encoder.fit_transform(data.loc['category_name'])
    data.to_csv('data/interim/extracted.csv', index=False, header=True, encoding='utf-8-sig')
    with open('label_encoder.pickle', 'wb') as file:
        pickle.dump(label_encoder, file)
