"""The module implements functions to preprocess data."""
import os.path as path

import pandas as pd
import pickle
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def encode_labels(data_path: str):
    """Accept path to dataset, saves dataset with encoded categories names.

    Args:
        data_path (str): path to dataset
    """
    data = pd.read_csv(data_path, encoding="utf-8-sig")
    label_encoder = LabelEncoder()
    data.loc[:, "category_name"] = label_encoder.fit_transform(
        data.loc[:, "category_name"]
    )
    three_up_path = path.abspath(path.join(__file__, "../../.."))
    output_path = three_up_path + "/data/interim/encoded_labels.csv"
    data.to_csv(output_path, index=False, header=True, encoding="utf-8-sig")
    print(f"Dataset with encoded labels saved to path: {output_path}")
    output_path = three_up_path + "/utils/label_encoder.pickle"
    with open(output_path, "wb") as file:
        pickle.dump(label_encoder, file)
    print(f"Labels encoder saved to path: {output_path}")


def clear_sentence(sentence: str) -> str:
    """Accept sentence, returns a clear sentence.

    Args:
        sentence (str): input sentence

    Returns:
        str: cleared sentence
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        re.UNICODE,
    )
    sentence = sentence.lower()  # to lower
    sentence = re.sub(emoji_pattern, "", sentence)  # delete emoji
    sentence = re.sub(r"\d\w{1,2}", "", sentence)  # delete digits
    sentence = re.sub(r"\d", "", sentence)  # delete digits
    sentence = re.sub(r"[^\w\s]", " ", sentence)  # delete symbols
    sentence = re.sub(r"[a-zA-Z\s]+", " ", sentence)  # delete english symbols
    return sentence


def clear_data(data_path: str):
    """Accept path to dataset, saves dataset with encoded categories names.

    Args:
        data_path (str): path to dataset
    """
    data = pd.read_csv(data_path, encoding="utf-8-sig")
    data.loc[:, "appeal_text"] = data.loc[:, "appeal_text"].apply(clear_sentence)
    three_up_path = path.abspath(path.join(__file__, "../../.."))
    output_path = three_up_path + "/data/processed/data.csv"
    data.to_csv(output_path, index=False, header=True, encoding="utf-8-sig")
    print(f"Cleared dataset saved to path: {output_path}")


def split(
    data_path: str, save: bool = False, random_state: int = 0
) -> tuple[list, list, list, list]:
    """Accept path to dataset, returns train and test data (80, 20 %).

    Args:
        data_path (str): path to dataset
        save (bool): saving train and eval set
        random_state (int): random state

    Returns:
        data, labels (tuple[list, list, list, list]): train, validation, test data and labels
    """
    data = pd.read_csv(data_path, encoding="utf-8-sig")
    texts = list(data.loc[:, "appeal_text"])
    labels = list(data.loc[:, "category_name"])
    train_data, test_data, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.20, random_state=random_state
    )
    if save:
        three_up_path = path.abspath(path.join(__file__, "../../.."))
        output_path = three_up_path + "/data/processed/train.csv"
        train_dataset = pd.DataFrame(
            zip(train_data, train_labels), columns=["appeal_text", "category"]
        )
        train_dataset.to_csv(
            output_path, index=False, header=True, encoding="utf-8-sig"
        )
        print(f"Train dataset saved to path: {output_path}")
        output_path = three_up_path + "/data/processed/test.csv"
        test_dataset = pd.DataFrame(
            zip(test_data, test_labels), columns=["appeal_text", "category"]
        )
        test_dataset.to_csv(output_path, index=False, header=True, encoding="utf-8-sig")
        print(f"Test dataset saved to path: {output_path}")
    return train_data, test_data, train_labels, test_labels


def reassign(path_to_data: str):
    """Reassign class labels

    Args:
        path_to_data (str): path_to_data
    """
    data = pd.read_csv(path_to_data, encoding="utf-8-sig")
    for i in range(data.shape[0]):
        text = data.iloc[i, 0]
        if "рекл" in text and "незак" not in text:
            data.iloc[i, 1] = 12
        elif "рекл" in text and "незак" in text:
            data.iloc[i, 1] = 6
        elif ("мусор" in text and "парад" in text) or (
            "мусор" in text and "подъезд" in text
        ):
            data.iloc[i, 1] = 10
        elif "кровл" in text or "крыша" in text:
            data.iloc[i, 1] = 3
        elif "торгов" in text:
            data.iloc[i, 1] = 7
        elif "информ" in text and "графф" not in text:
            data.iloc[i, 1] = 12
        elif "графф" in text and "нарк" in text:
            data.iloc[i, 1] = 6
        elif "подвал" in text:
            data.iloc[i, 1] = 9
        elif "фасад" in text:
            data.iloc[i, 1] = 13

        category = data.iloc[i, 1]
        if category in [1, 2, 14]:
            data.iloc[i, 1] = 1
        elif category in [4, 5]:
            data.iloc[i, 1] = 3
        elif category == 3:
            data.iloc[i, 1] -= 1
        elif category >= 6:
            data.iloc[i, 1] -= 2

    reassign_dict = {
        0: "Благоустройство",
        1: "Водоснабжение/отведение и отопление",
        2: "Кровля",
        3: "Нарушение порядка и правил пользования общим имуществом",
        4: "Незаконная информационная и (или) рекламная конструкция",
        5: "Незаконная реализация товаров с торгового оборудования (прилавок, ящик, с земли)",
        6: "Повреждения или неисправность элементов уличной инфраструктуры",
        7: "Подвалы",
        8: "Санитарное состояние",
        9: "Содержание МКД",
        10: "Состояние рекламных или информационных конструкций",
        11: "Фасад",
    }
    three_up_path = path.abspath(path.join(__file__, "../../.."))
    output_path = three_up_path + "/data/processed/data_reassign.csv"
    data.to_csv(output_path, index=False, header=True, encoding="utf-8-sig")
    print(f"Reassigned dataset saved to path: {output_path}")
    output_path = three_up_path + "/utils/reassign_dict.pickle"
    with open(output_path, "wb") as file:
        pickle.dump(reassign_dict, file)
    print(f"Reassign dict saved to path: {output_path}")
