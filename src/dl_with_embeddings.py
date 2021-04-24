# %% imports
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
import numpy as np
import pandas as pd
import pandarallel
from pandarallel import pandarallel
from sgt import SGT
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

# %%
embeddings_joined = pd.read_csv("embeddings_first_20000_Antibodies_kappa_1.csv")
# shuffle frame
embeddings_joined = embeddings_joined.sample(frac=1)
paired_embeddings = embeddings_joined.iloc[:round(embeddings_joined.shape[0]/2), :]
non_paired_embeddings = embeddings_joined.iloc[round(embeddings_joined.shape[0]/2):, :]
non_paired_embeddings = pd.concat([non_paired_embeddings.iloc[:,:401], non_paired_embeddings.iloc[:,401:].sample(frac=1)], axis=1)

# %%
numerical_columns = embeddings_joined.columns[1:]
# 1 = paired 0 = unpaired
output = [1] * round(embeddings_joined.shape[0]/2) + [0] * round(embeddings_joined.shape[0]/2)
