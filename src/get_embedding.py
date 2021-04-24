# %% imports
import numpy as np
import pandas as pd
import pandarallel
from pandarallel import pandarallel
from sgt import SGT


# %% ####
"""try to get embedding from original data"""
pandarallel.initialize(nb_workers=4)
data = pd.read_csv("antibody_pairing.csv").reset_index()
data['index'] = "paired_"+data['index'].astype(str)
# %%
data = data.loc[:, ["tenx_barcode", "sequence_heavy", "sequence_light",
                    "tenx_chain_heavy", "tenx_chain_light"]]
corpus_heavy = data.loc[:, ["tenx_barcode", "sequence_heavy"]]
corpus_heavy.columns = ["id", "sequence"]
corpus_light = data.loc[:, ["tenx_barcode", "sequence_light"]]
corpus_light.columns = ["id", "sequence"]
corpus_heavy['sequence'] = corpus_heavy['sequence'].map(list)
corpus_light['sequence'] = corpus_light['sequence'].map(list)
# process corpus to the right format (split sequences to list of chars)
corpus_heavy.head()
# %%
%%time
# Compute SGT embeddings
sgt_ = SGT(kappa=1,
           lengthsensitive=False,
           mode='multiprocessing')


# %%
# just take the first 1000 lines for testing
sgtembedding_light = sgt_.fit_transform(corpus_light.iloc)
# %%
sgtembedding_heavy = sgt_.fit_transform(corpus_heavy.iloc)

# %%
colnames = ["heavy_" + str(col) for col in sgtembedding_heavy.columns[1:]]
sgtembedding_heavy.columns = ["id"] + colnames
colnames = ["light_" + str(col) for col in sgtembedding_light.columns[1:]]
sgtembedding_light.columns = ["id"] + colnames

embeddings_joined = pd.concat([sgtembedding_heavy.set_index(
    "id"), sgtembedding_light.set_index("id")], axis=1)
embeddings_joined.to_csv("all_pairings.csv")
