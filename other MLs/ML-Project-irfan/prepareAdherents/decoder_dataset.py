import pandas as pd
import pickle

# Charger le dataset encodé
dataset_encoded = pd.read_csv('dataset_encoded.csv')

# Charger les encoders
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Décoder si nécessaire
dataset_decoded = dataset_encoded.copy()
for col in label_encoders.keys():
    if col in dataset_decoded.columns:
        dataset_decoded[col] = label_encoders[col].inverse_transform(dataset_decoded[col])