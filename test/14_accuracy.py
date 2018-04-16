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
with open ('../data/sync_25fps/common_point/result.json') as f_buff : 
    common_plane_coor_gt = json.load (f_buff)

# load json result data
with open ('result/500-iteration.json', 'r') as f_buff: 
    common_plane_coor_rs = json.load (f_buff)

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

# initialize zeros mask
shape = next (fi[view]).shape
img_gt = {}
img_rs = {}
for view in session[_id] : 
    img_gt[view] = np.zeros ((shape[:2]))
    img_rs[view] = np.zeros ((shape[:2]))

for i in range (300) : 
    view_frame = None
    for view in session[_id] : 
        # draw mask
        img_color = next (fi[view])
        # img_gt[view] = cv2.addWeighted (img_color, 1, masks[view], 0.0, 0)
        # img_rs[view] = cv2.addWeighted (img_color, 1, masks[view], 0.0, 0)

        """
        Ground Truth drawing
        """

        # line ground truth
        lines =  [ 
                Line.from_two_points (common_plane_coor_gt['session0'][view][0], vps[view]['vp2']),  # top
                Line.from_two_points (common_plane_coor_gt['session0'][view][-1], vps[view]['vp2']), # bottom
                Line.from_two_points (common_plane_coor_gt['session0'][view][1], vps[view]['vp1']), # right 
                Line.from_two_points (common_plane_coor_gt['session0'][view][2], vps[view]['vp1']), # left 
            ]

        # get point intersection gt
        rect_coor = [
                tuple ([int (__) for __ in lines[0].get_intersection (lines[-1])]), # top-left
                tuple ([int (__) for __ in lines[0].get_intersection (lines[-2])]), # top-right
                tuple ([int (__) for __ in lines[1].get_intersection (lines[-2])]),  # bot-right
                tuple ([int (__) for __ in lines[1].get_intersection (lines[-1])]) # bot-left
            ]

        # draw coor gt
        """
        for p_idx, p in enumerate (rect_coor) : 
            img[view] = cv2.circle (img[view], p , 10, (0,255, 0), -1)

            # put id
            loc = (p[0] + 10, p[1] - 20) 
            cv2.putText (img[view],  str (p_idx + 1), loc, cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 100), 4)
        """

        rect_coor = np.array (rect_coor, np.int32)
        rect_coor = rect_coor.reshape ((-1, 1, 2))
        cv2.fillConvexPoly (img_gt[view], rect_coor, color=(255,255,255))

        # """

        """
        Result drawing
        """

        # get point intersection result 
        rect_coor = common_plane_coor_rs['session0'][view]
        rect_coor = map ( lambda p : tuple ([int (__) for __ in p]), rect_coor)
        rect_coor = [
                rect_coor[0],
                rect_coor[1],
                rect_coor[3],
                rect_coor[2]
            ]

        # draw coor res
        """
        for p_idx, p in enumerate (rect_coor) : 
            img[view] = cv2.circle (img[view], p , 10, (0,0,255), -1)

            # put id
            loc = (p[0] + 10, p[1] - 20) 
            cv2.putText (img[view],  str (p_idx + 1), loc, cv2.FONT_HERSHEY_PLAIN, 3, (0, 100, 255), 4)
        """

        rect_coor = np.array (rect_coor, np.int32)
        rect_coor = rect_coor.reshape ((-1, 1, 2))
        cv2.fillConvexPoly (img_rs[view], rect_coor, color=(255,255,255))
        # cv2.polylines (img_rs[view], [rect_coor], True ,color=(0,0,255), thickness=5)
        # """

        """
        Convert into biner and do bitwise_and
        """
        biner_gt = cv2.threshold(img_gt[view] , 100, 255, cv2.THRESH_BINARY)[1]
        biner_rs = cv2.threshold(img_rs[view] , 100, 255, cv2.THRESH_BINARY)[1]

        intersection = cv2.bitwise_and (biner_gt, biner_rs)

        img_gt[view] =  biner_gt 
        overlap = np.sum (intersection) / np.sum (biner_gt)
        print ("View {} accuracy : {:.2F} %".format (view, overlap * 100))

        # combine view
        if view_frame is None : 
            view_frame = img_gt[view]
        else : 
            view_frame = np.hstack ((img_gt[view], view_frame))

    # add text
    loc = (20, view_frame.shape[0]-20)
    cv2.putText (view_frame, 'Frame - {}'.format (i + 1), loc, cv2.FONT_HERSHEY_PLAIN, 3, (0, 128, 128), 4)

    # show time
    cv2.imshow ('default', view_frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break
    
    if i == 60 : 
        for view in session[_id] : 
            cv2.imwrite ('result/gt-{}.jpg'.format (view), img_rs[view])
        sys.exit ()


