from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
import numpy as np
import pandas as pd
import pandarallel
from pandarallel import pandarallel
from sgt import SGT
import sys


# %%
"""try to get embedding from original data"""
pandarallel.initialize(nb_workers=10)
data = pd.read_csv("antibody_pairing.csv").reset_index()
data['index'] = "paired_"+data['index'].astype(str)

# %%
data = data.loc[:, ["index", "sequence_alignment_aa_heavy", "sequence_alignment_aa_light",
               "tenx_chain_heavy", "tenx_chain_light"]]
corpus_heavy = data.loc[:, ["index", "sequence_alignment_aa_heavy"]]
corpus_heavy.columns = ["id", "sequence"]
corpus_light = data.loc[:, ["index", "sequence_alignment_aa_light"]]
corpus_light.columns = ["id", "sequence"]
corpus_heavy['sequence'] = corpus_heavy['sequence'].map(list)
corpus_light['sequence'] = corpus_light['sequence'].map(list)
# process corpus to the right format (split sequences to list of chars)
corpus_heavy.head()
# %%
# Compute SGT embeddings
embeddings_joined = {}
slices = [(0,20000), (20000,40000), (40000,60000), (60000,80000), (80000,100000), (100000,127580)]
for slice_ in slices:
    sgt_ = SGT(kappa=5,
            lengthsensitive=False,
            mode='multiprocessing')
    
    # just take the first 1000 lines for testing
    sgtembedding_light = sgt_.fit_transform(corpus_light.iloc[slice_[0]:slice_[1]])
    
    sgtembedding_heavy = sgt_.fit_transform(corpus_heavy.iloc[slice_[0]:slice_[1]])

    
    colnames = ["heavy_" + str(col) for col in sgtembedding_heavy.columns[1:]]
    sgtembedding_heavy.columns = ["id"] + colnames
    colnames = ["light_" + str(col) for col in sgtembedding_light.columns[1:]]
    sgtembedding_light.columns = ["id"] + colnames

    embeddings_joined_tmp = pd.concat([sgtembedding_heavy.set_index(
        "id"), sgtembedding_light.set_index("id")], axis=1)
    embeddings_joined[slice_] = embeddings_joined_tmp


embeddings_joined_df = pd.concat([df for df in embeddings_joined.values()], axis=0)
embeddings_joined_df.to_csv("all_pairings.csv")

# %%
