"""The module implements data extraction function."""

import os.path as path
import pandas as pd
import os
import json


def extract_dataset(files_path: str):
    """Accept path to files with data, saves dataset with appeals, categories names.

    Args:
        files_path (str): path to files
    """
    files = os.listdir(files_path)
    rows = []
    for file in files:
        file_path = os.path.join(files_path, file)
        with open(file_path, encoding="utf-8", errors="ignore") as json_data:
            data = json.load(json_data)
            text = ""
            for widget in data["feed"]:
                if widget["widget"] == "public.petition":
                    for text_part in widget["payload"]["body"]:
                        if text_part["typeof"] == 1:
                            text += text_part["text"]
                    break

            category_name = data["reason"]["category"]["name"]
            rows.append([text, category_name])

    data = pd.DataFrame(rows, columns=["appeal_text", "category_name"])
    data = data.loc[:, ["appeal_text", "category_name"]]
    three_up_path = path.abspath(path.join(__file__, "../../.."))
    output_path = three_up_path + "/data/interim/extracted.csv"
    data.to_csv(output_path, index=False, header=True, encoding="utf-8-sig")
    print(f"Data extracted to path: {output_path}")
