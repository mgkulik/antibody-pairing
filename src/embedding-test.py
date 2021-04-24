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
# %%
# load toy data
# Loading uniprot data
corpus = pd.read_csv('prot_data_test.csv')

# Data preprocessing for uniprot data
corpus = corpus.loc[:, ['Entry', 'Sequence']]
corpus.columns = ['id', 'sequence']
corpus['sequence'] = corpus['sequence'].map(list)
corpus

# %%
%%time
# Compute SGT embeddings
sgt_ = SGT(kappa=1,
           lengthsensitive=False,
           mode='multiprocessing')
sgtembedding_df = sgt_.fit_transform(corpus)
# Set the id column as the dataframe index
sgtembedding_df = sgtembedding_df.set_index('id')
sgtembedding_df

# %% ####
"""try to get embedding from original data"""
pandarallel.initialize(nb_workers=4)
data = pd.read_csv("antibody_pairing.csv")
# %%
data = data.loc[:, ["tenx_barcode", "sequence_heavy", "sequence_light",
               "tenx_chain_heavy", "tenx_chain_light"]]
corpus_heavy = data.loc[:, ["tenx_barcode","sequence_heavy"]]
corpus_heavy.columns = ["id", "sequence"]
corpus_light = data.loc[:, ["tenx_barcode", "sequence_light"]]
corpus_light.columns = ["id", "sequence"]
corpus_heavy['sequence'] = corpus_heavy['sequence'].map(list)
corpus_light['sequence'] = corpus_light['sequence'].map(list)
# process corpus to the right format (split sequences to list of chars)
corpus_heavy.head()

# %%
# just take the first 1000 lines for testing
sgtembedding_light = sgt_.fit_transform(corpus_light.iloc[:10000,:])
# %%
sgtembedding_heavy = sgt_.fit_transform(corpus_heavy.iloc[:10000,:])

# %%
meta_data = data.loc[:9999, ["tenx_barcode", "tenx_chain_heavy", "tenx_chain_light"]]

# %%
# join meta and the two embeddings

embeddings_joined = pd.concat([sgtembedding_heavy.set_index(
    "id"), sgtembedding_light.set_index("id")], axis=0)
# run PCA and cluster
pca = PCA(n_components=2)
pca.fit(embeddings_joined)

X = pca.transform(embeddings_joined)

print(np.sum(pca.explained_variance_ratio_))
df = pd.DataFrame(data=X, columns=['x1', 'x2'])
df.shape

#%% plot chain types on pca

types = meta_data.tenx_chain_heavy.to_list() + meta_data.tenx_chain_light.to_list()

fig = plt.figure(figsize=(5, 5))
colmap = {'IGH': 'r', 'IGK': 'g', 'IGL': 'b'}
#colors = list(map(lambda x: colmap[x+1], types))
sns.scatterplot(df['x1'], df['x2'], hue=types, alpha=0.5)

# %%
sgtembedding_heavy_out = sgtembedding_heavy.copy()
sgtembedding_light_out = sgtembedding_light.copy()

colnames = ["heavy_" + str(col) for col in sgtembedding_heavy_out.columns[1:]]
sgtembedding_heavy_out.columns = ["id"] + colnames
colnames = ["light_" + str(col) for col in sgtembedding_light_out.columns[1:]]
sgtembedding_light_out.columns = ["id"] + colnames

embeddings_joined_out = pd.concat([sgtembedding_heavy_out.set_index(
    "id"), sgtembedding_light_out.set_index("id")], axis=1)
#embeddings_joined_out.to_csv("embeddings_first_1000_Antibodies.csv")
# %%
