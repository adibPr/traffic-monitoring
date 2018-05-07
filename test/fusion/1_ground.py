#!/usr/bin/env python2
from __future__ import print_function, division
import os
import sys
import json

import cv2
import numpy as np

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..', '..'))

from geometry import get_extreme_tan_point, Line, get_extreme_side_point, find_right_most_point
from util import *
from iterator import FrameIterator

def draw_polylines (img, corner, color=(0,0,255)) : 
    img = img.copy ()

    cornerInt = []
    for c in corner : 
        cornerInt.append (tuple ([int (_) for _ in c]))

    corner = np.array (corner, np.int32)
    corner = corner.reshape ((-1, 1, 2))

    cv2.polylines (img, [corner], True , color, thickness=5)
    return img

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

VP = VPLoader ()

VIEW = ("right", "center", "left")
with open ("/home/adib/My Git/traffic-monitoring/data/sync_25fps/common_point/result.json", "r") as f_buf : 
    GT = json.load (f_buf)

cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)
img_path = '/home/adib/My Git/traffic-monitoring/data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/screen.png'

ses_id = 5 
vp = VP.get_session (ses_id)
fi = FrameIteratorLoader.get_session (ses_id)

M = {}
ts_img = {}
prev_img = {}
for view in VIEW : 
    img = cv2.imread (img_path.format (ses_id, view), 1)

    points = GT['session{}'.format (ses_id)][view]
    corner = get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points)

    # get rectangular homography mapping
    corner_gt = np.float32 (corner)
    corner_wrap = np.float32 ([[0,300],[0,0], [1000,0], [1000, 300]])
    M[view] = cv2.getPerspectiveTransform (corner_gt, corner_wrap)

    # for initialization
    ts_img[view] = None # np.zeros ((300, 1000, 3)) 

    # for 3 frame difference
    prev_img[view] = [None, None]
    for i in range (2) : 
        img = next (fi[view])
        img_color = img.copy ()
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

        # save background
        prev_img[view][i] = img 

size = 5 
for i in range (300) : 
    frame = None
    for view in VIEW : 
        img_color = next (fi[view])
        img = cv2.cvtColor (img_color, cv2.COLOR_BGR2GRAY)

        # by 3 frame difference
        prev_intersect = cv2.threshold (cv2.absdiff (prev_img[view][1], prev_img[view][0]), 25, 255, cv2.THRESH_BINARY)[1]
        next_intersect = cv2.threshold (cv2.absdiff (img, prev_img[view][1]), 25, 255, cv2.THRESH_BINARY)[1]

        # by 3 frame of "Vehicle speed measurement based on gray constraint optical flow algorithm"
        P1 = cv2.bitwise_and (prev_intersect, next_intersect)
        # prev_intersect_dilate = cv2.dilate (prev_intersect, kernel10)
        # next_intersect_dilate = cv2.dilate (next_intersect, kernel10)
        prev_intersect_dilate = process_morphological (prev_intersect) 
        next_intersect_dilate = process_morphological (next_intersect)
        
        P2 = cv2.bitwise_and (prev_intersect_dilate, next_intersect_dilate)
        # fg_3frame = cv2.bitwise_xor (P1, P2)

        dst = cv2.warpPerspective (img_color, M[view], (1000,300))
        
        # draw a middle line
        # cv2.line (dst, (700, 0), (700, 300), color=(255, 255, 0), thickness=10)
        border_img = dst[:, 700:700+size]

        # time spatial construction
        if ts_img[view] is None : 
            ts_img[view] = border_img
        else : 
            ts_img[view] = np.hstack ((ts_img[view], border_img))

        if ts_img[view].shape[1] > 1000 : 
            ts_img[view] = ts_img[view][:, size:]

        if frame is None : 
            frame = ts_img[view] 
        else : 
            frame = np.vstack ((frame, ts_img[view]))

        prev_img[view][0] = prev_img[view][1]
        prev_img[view][1] = img


    print (frame.shape)
    cv2.imshow ('default', frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break
