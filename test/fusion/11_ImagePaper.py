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

VP = VPLoader ()

VIEW = ("right", "center", "left")
with open ("/home/adib/My Git/traffic-monitoring/data/sync_25fps/common_point/result.json", "r") as f_buf : 
    GT = json.load (f_buf)

cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)
img_path = '/home/adib/My Git/traffic-monitoring/data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/screen.png'

ses_id = 5 
vp = VP.get_session (ses_id)
fi = FrameIteratorLoader.get_session (ses_id)

M = {} # matrix homography
ts_img = {} # initalization of time spatial image
epi_img = {} # initialization of epi image
prev_img = {} # prev 2 image for 3 frame differance
prev_tsi = {} # prev 2 image for 3 frame difference time spatial image
bms = {} # background model MoG
VDL_IDX = 100 # column index of VDL
VDL_SIZE = 5  # size of VDL

for view in VIEW : 
    img = cv2.imread (img_path.format (ses_id, view), 1)

    points = GT['session{}'.format (ses_id)][view]
    corner = get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points)

    # get rectangular homography mapping
    corner_gt = np.float32 (corner)
    corner_wrap = np.float32 ([[0,300],[0,0], [1000,0], [1000, 300]])
    M[view] = cv2.getPerspectiveTransform (corner_gt, corner_wrap)

    # background subtraction
    bms[view] = BackgroundModel (fi[view], detectShadows=False)
    bms[view].learn (tot_frame_init=2)

    # for initialization
    ts_img[view] = None
    epi_img[view] = None

    # for 3 frame difference
    prev_img[view] = [None, None]
    for i in range (2) : 
        img = next (fi[view])
        img_color = img.copy ()
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

        # save background
        prev_img[view][i] = img 

    prev_tsi[view] = [None, None]

while True:

    frame = None
    disp_center = None
    disp_track = None

    for view in VIEW : 
        img_color = next (fi[view])
        img = cv2.cvtColor (img_color, cv2.COLOR_BGR2GRAY)

        # for displyaing background result
        # dst = cv2.warpPerspective (P2, M[view], (1000,300))
        
        dst = cv2.warpPerspective (img, M[view], (1000, 300))

        if view == 'center' : 
            disp_center = dst.copy ()
            # display VDL
            cv2.line (disp_center, (VDL_IDX, 0), (VDL_IDX, dst.shape[1]), color=(255, 255, 255), thickness=10)
            # display tracking line
            cv2.line (disp_center, (0, 75), (1000, 75), color=(255, 255, 255), thickness=10)

        # take only the LINE_COL column
        border_img = dst[:, VDL_IDX:VDL_IDX+VDL_SIZE]

        # time spatial construction
        if ts_img[view] is None : 
            ts_img[view] = border_img
        else : 
            ts_img[view] = np.hstack ((border_img, ts_img[view]))

        cur_disp = ts_img[view]
        if ts_img[view].shape[1] > 1000 : 
            ts_img[view] = ts_img[view][:, :-VDL_SIZE]

            if all ([_ is not None for _ in prev_tsi[view]]) :  # only when all frame is 1000 already

                # by 3 frame of "Vehicle speed measurement based on gray constraint optical flow algorithm"
                prev_intersect = cv2.threshold (cv2.absdiff (prev_tsi[view][1], prev_tsi[view][0]), 25, 255, cv2.THRESH_BINARY)[1]
                next_intersect = cv2.threshold (cv2.absdiff (ts_img[view], prev_tsi[view][1]), 25, 255, cv2.THRESH_BINARY)[1]
                P1 = cv2.bitwise_and (prev_intersect, next_intersect)
                prev_intersect_dilate = process_morphological (prev_intersect) 
                next_intersect_dilate = process_morphological (next_intersect)
                P2 = cv2.bitwise_and (prev_intersect_dilate, next_intersect_dilate)

                cur_disp = P2 

            # update 3FD for tsi
            prev_tsi[view][0] = prev_tsi[view][1]
            prev_tsi[view][1] = ts_img[view] 

        # EPI construction
        epi_part = np.transpose (dst[75:75+VDL_SIZE, :1000])
        if epi_img[view] is None : 
            epi_img[view] = epi_part
        else : 
            epi_img[view] = np.hstack ((epi_part, epi_img[view]))

        if epi_img[view].shape[1] > 1000 : 
            epi_img[view] = epi_img[view][:, :1000]
        disp_track = epi_img[view]

        if frame is None : 
            frame = cur_disp
            intersection = cur_disp

        else : 
            frame = np.vstack ((frame, cur_disp))
            intersection = cv2.bitwise_and (intersection, cur_disp)

        # prev_img[view][0] = prev_img[view][1]
        # prev_img[view][1] = img

    # start when everything is done
    if frame.shape[1] != 1000 : 
        continue


    # add morphological process to intersection
    intersection= process_morphological (intersection)

    # concat view
    # frame = np.vstack ((frame, intersection))
    frame = np.vstack ((disp_center, intersection, epi_img['center']))
    cv2.imshow ('default', frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break

