#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np
import os
import sys
from numpy.linalg  import norm
from scipy.spatial.distance import cosine

GT_LIST_ID = [0,1,2,4,5,6,8,13,14,15,16,17,18,19,21,24,25,26,27,28,29,31,32,34,35,36,37,38,40,41,42,43,44,45,46,47,51]
CLIP_ID = [0, 1, 13, 18, 26, 31]
CLIP_IDX = [ idx  for (idx,_id) in enumerate (GT_LIST_ID) if _id in CLIP_ID ] # list index of selected ID

filename_data = [ d for d in os.listdir (os.path.abspath (os.path.dirname (__file__))) if d.endswith ('.npy') ]
data = {}
for d in filename_data : 
    data[d[:-4]] = np.load (d)

y = data['gt_all'] == 0

for key in data :
    truth_table = data[key] == data['gt_all']
    truth_per_row = []

    for i in range (truth_table.shape[1]) : 
        value, counts = np.unique (truth_table[:,i], return_counts=True)
        truth_per_row.append (dict (zip (value, counts)))

    print ("For data in {}".format (key))
    for idx in CLIP_IDX : 
        clip_id = GT_LIST_ID[idx]
        print (" -  {} : {:.2F}".format (clip_id, truth_per_row[idx][True] / sum (truth_per_row[idx].values ()) * 100 ))
        gt = np.array (data['gt_all'][:,idx])
        res = np.array (data[key][:,idx])
        # print ("  - {} : By norm : {:.2F}".format (clip_id, 1 - norm (gt-res )))
