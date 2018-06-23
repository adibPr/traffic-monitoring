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
from background import BackgroundModel

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

def TSI_get_contours (frame_binary, min_area=200, min_width=70) :
    contours_selected = []

    im2, contours, hierarchy = cv2.findContours(
            frame_binary, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )

    # select contour that greater than the specified $min_area and
    # not part of other contour (specified by its hierarchy)
    # additionally, its width must be greater than $min_width
    for c_idx, c in enumerate (contours) :
        (x,y, w, h) = cv2.boundingRect (c)
        if cv2.contourArea(c) > min_area and hierarchy[0][c_idx][3] < 0 and \
                h > min_width:
            contours_selected.append (c)

    return contours_selected

VP = VPLoader ()

VIEW = ("right", "center", "left")
with open ("/home/adib/My Git/traffic-monitoring/data/sync_25fps/common_point/result.json", "r") as f_buf : 
    GT = json.load (f_buf)

# cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)
img_path = '/home/adib/My Git/traffic-monitoring/data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/screen.png'

ses_id = 0 
vp = VP.get_session (ses_id)
fi = FrameIteratorLoader.get_session (ses_id)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('result/TSI_video/ses_{}.avi'.format (ses_id),fourcc, 25.0, (1000, 900))

M = {} # matrix homography
ts_img = {} # initalization of time spatial image
prev_img = {} # prev 2 image for 3 frame differance
prev_tsi = {} # prev 2 image for 3 frame difference time spatial image
bms = {} # background model MoG

for view in VIEW : 
    img = cv2.imread (img_path.format (ses_id, view), 1)

    points = GT['session{}'.format (ses_id)][view]
    corner = get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points)

    # get rectangular homography mapping
    corner_gt = np.float32 (corner)
    corner_wrap = np.float32 ([[0,300],[0,0], [1000,0], [1000, 300]])
    M[view] = cv2.getPerspectiveTransform (corner_gt, corner_wrap)
    print (M[view].shape)
    sys.exit ()

    # background subtraction
    bms[view] = BackgroundModel (fi[view], detectShadows=False)
    bms[view].learn (tot_frame_init=2)

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

    prev_tsi[view] = [None, None]

size = 5 
center_list = [] # for tracking
T_DIST = 4 # threshold distance
T_XRANGE = 20 # threshold of consideration of out vehicle
tot_vehicle = 0
ctr = 0
try : 
    while True:
        ctr += 1
        frame = None
        for view in VIEW : 
            img = img_color = next (fi[view])
            # img = cv2.cvtColor (img_color, cv2.COLOR_BGR2GRAY)

            # for displyaing background result
            # dst = cv2.warpPerspective (P2, M[view], (1000,300))
            
            dst = cv2.warpPerspective (img, M[view], (1000, 300))

            # take only the 700 column
            border_img = dst[:, 700:700+size, :]

            # time spatial construction
            if ts_img[view] is None : 
                ts_img[view] = border_img
            else : 
                ts_img[view] = np.hstack ((border_img, ts_img[view]))


            if ts_img[view].shape[1] > 1000 : 
                ts_img[view] = ts_img[view][:, :1000, :]
            else : 
                needed_shape = 1000 - ts_img[view].shape[1]
                padding = np.zeros ((300, needed_shape, 3)).astype ('uint8')
                ts_img[view] = np.hstack ((padding, ts_img[view]))

            cur_disp = ts_img[view]


            if frame is None : 
                frame = cur_disp

            else : 
                frame = np.vstack ((frame, cur_disp))

            # prev_img[view][0] = prev_img[view][1]
            # prev_img[view][1] = img

        # start when everything is done
        # if frame.shape[1] != 1000 : 
        #     continue

        # add morphological process to intersection
        # cv2.imshow ('default', frame)
        out.write (frame)
        print (ctr)
        if (cv2.waitKey(1) & 0xFF == ord('q')) :
            out.release ()
            break

except Exception as e : 
    out.release ()
