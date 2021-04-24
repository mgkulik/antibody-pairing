#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 07:23:00 2021

@author: magoncal
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data_embeddings = pd.read_csv('embeddings_first_1000_Antibodies.csv').reset_index()
data_embeddings['index'] = "paired_"+data_embeddings['index'].astype(str)




