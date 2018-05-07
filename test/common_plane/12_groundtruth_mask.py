#!/usr/bin/env python

from __future__ import print_function, division
import os
import sys

import cv2
import numpy as np

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..'))

from iterator import FrameIterator
import util

mask = util.MaskLoader ()

for ses_id in range (0, 7) : 
    # load masks
    this_mask =  mask.get_session (ses_id, as_color=True)

    # load frame iterator
    fi = util.FrameIteratorLoader.get_session (ses_id)

    for view in this_mask.keys () : 
        img_color = next (fi[view])
        img_color = cv2.addWeighted (img_color, 0.7, this_mask[view], 0.3, 0)

        # check output directory if exist
        if not os.path.exists ('result/masked') : 
            os.makedirs ('result/masked')

        cv2.imwrite ('result/masked/session{}-{}.jpg'.format (ses_id, view), img_color)
