"""The file runs commands to prepare the data."""
from src.features.extract_data import extract_dataset
from src.features.preprocess import encode_labels, clear_data

extract_dataset(r'D:\Projects\Appeals-to-Housing-and-Communal-services\data\raw\src_files')
encode_labels(r'D:\Projects\Appeals-to-Housing-and-Communal-services\data\interim\extracted.csv')
clear_data(r'D:\Projects\Appeals-to-Housing-and-Communal-services\data\interim\encoded_labels.csv')
