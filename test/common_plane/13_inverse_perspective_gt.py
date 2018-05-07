#!/usr/bin/env python

from __future__ import print_function, division
import os
import sys
import json

import cv2
import numpy as np

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..'))


from iterator import FrameIterator
from geometry import get_extreme_tan_point, Line, get_extreme_side_point, find_right_most_point

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

# load json gt data
with open ('../data/sync_25fps/common_point/result.json') as f_buf : 
    common_plane_coor = json.load (f_buf)

# load vanishing points
vps = {}
for view in session[_id] : 
    vp_path = '../data/gt/2016-ITS-BrnoCompSpeed/results/session{}_{}/system_dubska_bmvc14.json'.format (_id, view)

    with open (vp_path, 'r') as f_buff :
        """
        vp has 2 keys
        - cars, list of cars detected, its frame and posX and posY
        - camera_calibration, the calibration parameter result (pp, vp1, and vp2)
        """
        vp = json.load (f_buff)
        vps[view] = {
                'vp1' : vp['camera_calibration']['vp1'],
                'vp2' : vp['camera_calibration']['vp2']
            }

img = {}
for i in range (300) : 
    view_frame = None
    for view in session[_id] : 
        # draw mask
        img_color = next (fi[view])
        img[view] = cv2.addWeighted (img_color, 0.7, masks[view], 0.3, 0)

        # draw line
        lines =  [ 
                Line.from_two_points (common_plane_coor['session0'][view][0], vps[view]['vp2']),  # top
                Line.from_two_points (common_plane_coor['session0'][view][-1], vps[view]['vp2']), # bottom
                Line.from_two_points (common_plane_coor['session0'][view][1], vps[view]['vp1']), # right 
                Line.from_two_points (common_plane_coor['session0'][view][2], vps[view]['vp1']), # left 
            ]

        for l in lines : 
            img[view] = l.draw (img[view])

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

