import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

np.random.seed(123)
torch.manual_seed(123)

#######################################
## Data manipulation and transformation
df = pd.read_csv("xx_projetos/alzheimer/Alzheimer.csv")
df = df.dropna(axis=0).reset_index(drop=True)
np_array_dataset = df.to_numpy()

# OneHotEnconder
one_hot_enconder = ColumnTransformer(transformers=[('OneHot',
                                               OneHotEncoder(),
                                               [0])],
                                               remainder='passthrough')
np_array_dataset = one_hot_enconder.fit_transform(np_array_dataset)

# Encoder - Just Column 4 "M/F"
encoder = LabelEncoder()
np_array_dataset[:, 3] = encoder.fit_transform(np_array_dataset[:, 3])

df = pd.DataFrame(np_array_dataset)
df.to_csv('alzheimer_formated.csv', index=False, encoding='utf-8')
