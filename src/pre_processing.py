#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 09:14:43 2021

@author: magoncal
"""

import pandas as pd
import numpy as np
from  glob  import  glob
import random

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

#data = pd.concat([pd.read_csv(file , header=1) for  file in glob('csv/*/*')])
#data.to_csv('antibody_pairing.csv', index=False)
data = pd.read_csv('antibody_pairing.csv').reset_index()
data['index'] = "paired_"+data['index'].astype(str)

# Generating short sample to test the model with anarci alignment
random.seed(12345)
class_ids = random.sample(range(0, len(data)), 100)

data_heavy = data.loc[class_ids, ['index', 'sequence_alignment_aa_heavy']].values.tolist()
data_light = data.loc[class_ids, ['index', 'sequence_alignment_aa_light']].values.tolist()

heavy_lst = []
i=1
for item in data_heavy:
    heavy_lst.append(SeqRecord(Seq(item[1]), id=str(item[0]), name=str(item[0]), description=""))
    if i%50==0 and i>0:
        with open('heavy_chain_'+str(i)+'.fasta', 'w') as handle:
            SeqIO.write(heavy_lst, handle, "fasta")
        heavy_lst = []
    i+=1

light_lst = []
i=1
for item in data_light:
    light_lst.append(SeqRecord(Seq(item[1]), id=str(item[0]), name=str(item[0]), description=""))
    if i%50==0 and i>0:
        with open('light_chain_'+str(i)+'.fasta', 'w') as handle:
            SeqIO.write(light_lst, handle, "fasta")
        light_lst = []
    i+=1
    
#Loading anarci data
anarci_heavy = pd.concat([pd.read_csv(file) for file in glob('anarci/*heavy*')])
anarci_light = pd.concat([pd.read_csv(file) for file in glob('anarci/*light*')])

cols = ["Id"]+list(anarci_heavy.columns)[-145:]
anarci_heavy = anarci_heavy.loc[:, cols]
for category in cols[1:]:
    anarci_heavy[category] = anarci_heavy[category].astype('category')

cols = ["Id"]+list(anarci_light.columns)[-127:]
anarci_light = anarci_light.loc[:, cols]
for category in cols[1:]:
    anarci_light[category] = anarci_light[category].astype('category')

data_filter = np.merge()