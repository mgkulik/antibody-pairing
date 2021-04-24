import argparse, joblib
import numpy as np
import pandas as pd

import extra_funcs

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from pandarallel import pandarallel

# set up argument parsing (make sure these match those in config.yml)
parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, required=True)
args = parser.parse_args()

# READ DATA
data = pd.read_csv(args.infile)

# embed both protein sequences respectively
sgt_ = SGT(kappa=5,
            lengthsensitive=False,
            mode='multiprocessing')

ids = ["ab_pair_{}".format(i) for i in range(data.shape[1])]
data["id"] = ids
heavy_embedding = sgt_.fit_transform(data[["id", "Hchain"]])
light_embedding = sgt_.fit_transform(data[["id", "Lchain"]])
#input transformed shoul have 800 cols containing 400 heavy and 400 light chains.
input_transformed = pd.concat([heavy_embedding.set_index("id"), light_embedding.set_index("id")], axis=1)


# PREDICT
modelfile = 'finalized_model.sav'
loaded_model = joblib.load(modelfile)
y_pred = loaded_model.predict_proba(input_transformed)

# SAVE PREDICTIONS WITH THE COLUMN NAME prediction IN THE FILE predictions.csv
pd.DataFrame(y_pred[:, 1], columns=['prediction']).to_csv("predictions.csv", index=False)
