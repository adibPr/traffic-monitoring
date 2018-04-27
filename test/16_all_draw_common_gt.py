#!/usr/bin/env python2
#!/usr/bin/env python

from __future__ import print_function, division
import os
import sys
import json

import cv2
import numpy as np

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..'))

from geometry import get_extreme_tan_point, Line, get_extreme_side_point, find_right_most_point
from util import *
from iterator import FrameIterator

def get_corner_ground (vp1, vp2, points) : 
    # convention of points : 
    #   [left, top, right, bottom]

    lines = [
            Line.from_two_points (vp1, points[0]), # left line
            Line.from_two_points (vp2, points[1]), # top line,
            Line.from_two_points (vp1, points[2]), # right line
            Line.from_two_points (vp2, points[3]) # bottom line
        ]

    corner = (
            lines[0].get_intersection (lines[1]), # top left corner
            lines[1].get_intersection (lines[2]), # top right corner
            lines[2].get_intersection (lines[3]), # bottom right corner
            lines[3].get_intersection (lines[0]) # bottom left corner
        )
    
    return corner

def draw_polylines (img, corner, color=(0,0,255)) : 
    img = img.copy ()

    cornerInt = []
    for c in corner : 
        cornerInt.append (tuple ([int (_) for _ in c]))

    corner = np.array (corner, np.int32)
    corner = corner.reshape ((-1, 1, 2))

    cv2.polylines (img, [corner], True , color, thickness=5)
    return img

def get_overlap (shape, pointsGT, pointsRS) : 
    # ground truth
    img_gt = np.zeros (shape[:2])
    pointsGT = np.array (pointsGT, np.int32)
    pointsGT = pointsGT.reshape ((-1, 1, 2))
    cv2.fillConvexPoly (img_gt, pointsGT, color=(255,255,255))

    # result
    img_rs = np.zeros (shape[:2])
    pointsRS = np.array (pointsRS, np.int32)
    pointsRS = pointsRS.reshape ((-1, 1, 2))
    cv2.fillConvexPoly (img_rs, pointsRS, color=(255, 255, 255))

    # convert into biner
    biner_gt = cv2.threshold(img_gt, 100, 255, cv2.THRESH_BINARY)[1]
    biner_rs = cv2.threshold(img_rs, 100, 255, cv2.THRESH_BINARY)[1]

    # intersection
    intersection = cv2.bitwise_and (biner_gt, biner_rs)

    # overlap
    overlap = np.sum (intersection) / np.sum (biner_gt)

    return overlap

VP = VPLoader ()

VIEW = ("right", "center", "left")
with open ("/home/adib/My Git/traffic-monitoring/data/sync_25fps/common_point/result.json", "r") as f_buf : 
    GT = json.load (f_buf)

with open ('result/4.json', 'r') as f_buf : 
    HS = json.load (f_buf)
    print (HS.keys ())

cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)
img_path = '/home/adib/My Git/traffic-monitoring/data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/screen.png'

for ses_id in range (0, 7) : 

    # some variable
    vp = VP.get_session (ses_id)
    fi = FrameIteratorLoader.get_session (ses_id)
    for view in VIEW : 
        # img = next (fi[view])
        img = cv2.imread (img_path.format (ses_id, view), 1)
        img_combine = img.copy ()

        """
        Ground Truth Section
        """
        points = GT['session{}'.format (ses_id)][view]
        corner = get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points)

        img_gt = draw_polylines (img, corner)
        img_combine = draw_polylines (img_combine, corner, color=(0,255,0))

        # cv2.imwrite ('result/common_ground_gt/{}-{}.jpg'.format (
        #     ses_id,
        #     view
        # ), img_gt )

        """ Result Section """
        if 'session{}'.format (ses_id) not in HS.keys () : 
            continue
        points = HS['session{}'.format (ses_id)][view]

        img_res = draw_polylines (img, points)
        img_combine = draw_polylines (img_combine, points, color=(0,0,255))

        overlap = get_overlap (img.shape, corner, points)
        print ("{}-{} overlap : {:.2F}".format (ses_id, view, overlap))

        # cv2.imwrite ('result/common_ground_gt/res-{}-{}.jpg'.format (
        #     ses_id,
        #     view
        # ), img_res )

        cv2.imwrite ('result/common_ground_gt/com-{}-{}.jpg'.format (
            ses_id,
            view
        ), img_combine )


