#!/usr/bin/env python

from __future__ import print_function, division
import os
import sys

import cv2
import numpy as np

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..'))


from iterator import FrameIterator

cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)

session = {
    0 : {
        'center' : None,
        'right' : None,
        'left' : None 
    }
}

_id = 0
# load masks
masks =  {}
for view in session[_id] : 
    mask_path = '../data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/video_mask.png'.format (_id, view)
    masks[view] = cv2.imread (mask_path, 0)
    masks[view] = cv2.cvtColor (masks[view], cv2.COLOR_GRAY2BGR) 
    

# generate frame iterator
fi = {}
for view in session[_id] : 
    fi[view] = FrameIterator ('../data/sync_25fps/session{}_{}'.format (_id, view))

img = {}
for i in range (300) : 
    view_frame = None
    for view in session[_id] : 
        img_color = next (fi[view])
        img[view] = cv2.addWeighted (img_color, 0.7, masks[view], 0.3, 0)

        # combine view
        if view_frame is None : 
            view_frame = img[view]
        else : 
            view_frame = np.hstack ((img[view], view_frame))

    # add text
    loc = (20, view_frame.shape[0]-20)
    cv2.putText (view_frame, 'Frame - {}'.format (i + 1), loc, cv2.FONT_HERSHEY_PLAIN, 3, (0, 128, 128), 4)

    # show time
    cv2.imshow ('default', view_frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break
    
    if i == 0 : 
        for view in session[_id] : 
            cv2.imwrite ('result/masked_bot-{}.jpg'.format (view), img[view])
