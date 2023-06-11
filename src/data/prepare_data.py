"""The file runs commands to prepare the data."""
import os.path as path

from src.features.extract_data import extract_dataset
from src.features.preprocess import encode_labels, clear_data, split, reassign
from src.features.cls_features import save_cls_features

three_up_path = path.abspath(path.join(__file__, "../../.."))
# extract_dataset(three_up_path + r"\data\raw\src_files")
# encode_labels(three_up_path + r"\data\interim\extracted.csv")
# clear_data(three_up_path + r"\data\interim\encoded_labels.csv")
# split(three_up_path + r"\data\processed\data.csv", save=True)
reassign(three_up_path + r"\data\processed\data.csv")
# path_to_model = "/models/bert_weighted.pt"
# if path.exists(three_up_path + path_to_model):
#     save_cls_features(
#         three_up_path + path_to_model,
#         path_to_save=three_up_path + "/data/processed/cls_features.csv",
#         path_to_data=three_up_path + "/data/processed/data.csv",
#     )
