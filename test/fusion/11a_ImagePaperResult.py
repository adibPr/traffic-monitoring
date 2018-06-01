#!/usr/bin/env python2
from __future__ import print_function, division
import os
import sys
import json

import cv2
import numpy as np
from pprint import pprint

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..', '..'))

from geometry import get_extreme_tan_point, Line, get_extreme_side_point, find_right_most_point
from util import *
from iterator import FrameIterator
from background import FrameDifference
import TSIUtil

VP = VPLoader ()

VIEW = ("right", "center", "left")
with open ("/home/adib/My Git/traffic-monitoring/data/sync_25fps/common_point/result.json", "r") as f_buf : 
    GT = json.load (f_buf)

cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)
img_path = '/home/adib/My Git/traffic-monitoring/data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/screen.png'

ses_id = 0 
vp = VP.get_session (ses_id)
fi = FrameIteratorLoader.get_session (ses_id)
# fi = {}

M = {} # matrix homography
M_inv = {} # inverse matrix homography
imgs_color = {} # for saving image color each view
dsts_color = {}
dsts = {}
fgs = {}
fdiff_tsi = {} # frame difference tsi image
fdiff_view = {} # frame difference camera image
tsi_object  = {}
posision_x_gt = [980, 758, 711]
posision_x_gt = posision_x_gt[::-1]
tsis = {}

for view in VIEW : 
    img = cv2.imread (img_path.format (ses_id, view), 1)
    
    # fi[view] = FrameIterator ('/home/adib/My Git/traffic-monitoring/test/fusion/result/sync_25fps_paper/{}'.format (view)) 

    points = GT['session{}'.format (ses_id)][view]
    corner = TSIUtil.get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points)

    # get rectangular homography mapping
    corner_gt = np.float32 (corner)
    corner_wrap = np.float32 ([[0,300],[0,0], [1000,0], [1000, 300]])
    M[view] = cv2.getPerspectiveTransform (corner_gt, corner_wrap)
    M_inv[view] = cv2.getPerspectiveTransform (corner_wrap, corner_gt)

    # initialize tsi object
    tsi_object[view] = TSIUtil.TSI (M[view], VDL_IDX=0)

    # for 3 frame difference
    prev_img = [None, None]
    prev_tsi = [None, None]
    for i in range (2) : 
        img = next (fi[view])
        # img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

        # save background
        prev_img[i] = img 

        # save tsi
        prev_tsi[i] = tsi_object[view].apply (img)

    fdiff_view[view] = FrameDifference (*prev_img)
    fdiff_tsi[view] = FrameDifference (*prev_tsi)

ctr = 0
while True:
    ctr += 1
    print (ctr)
    disp_tsi = None
    intersection = None
    disp_img = None

    for view in VIEW : 
        img = next (fi[view])
        imgs_color[view] = img.copy ()
        # img = cv2.cvtColor (img_color, cv2.COLOR_BGR2GRAY)

        dst = cv2.warpPerspective (img, M[view], (1000, 300))
        dsts[view] = dst

        line = [[0, 0], [0, 300]]
        img_line = TSIUtil.map_point (line, M_inv[view])
        cv2.line (imgs_color[view], tuple (img_line[0]), tuple (img_line[1]), color=(255,255,0), thickness=3)

        tsi = tsi_object[view].apply (img)
        tsi_fg = fdiff_tsi[view].apply (tsi)
        tsis[view] = tsi
        
        # fg = fdiff_view[view].apply (img, iterations=2)
        # fgs[view] = fg

        if disp_tsi is None : 
            disp_tsi = tsi
            intersection = tsi_fg
            disp_img = imgs_color[view]
        else : 
            disp_tsi = np.vstack ((tsi, disp_tsi))
            intersection = cv2.bitwise_and (intersection, tsi_fg)
            disp_img = np.vstack ((disp_img, imgs_color[view]))

        # cv2.imwrite ('result/sync_25fps_paper/{}/{}.jpg'.format (view, str (ctr).zfill (4)), img_color)

    """
    blobs = TSIUtil.get_contours (intersection)
    blobs = TSIUtil.get_most_left_blobs (blobs, n=3)
            
    intersection = cv2.cvtColor (intersection, cv2.COLOR_GRAY2BGR)
    intersection = draw_bounding_box_contours (intersection, blobs)
    disp_tsi = cv2.cvtColor (disp_tsi, cv2.COLOR_GRAY2BGR)
    dst = cv2.cvtColor (dst, cv2.COLOR_GRAY2BGR)
    disp = np.vstack ((disp_tsi, intersection, dst))
    """

    cv2.imshow ('default', disp_tsi)

    if ctr >= 200 and ctr <= 270 : 
        cv2.imwrite ('result/TSI_generation/view_{}.jpg'.format (ctr), imgs_color['center'])
        cv2.imwrite ('result/TSI_generation/TSI_{}.jpg'.format (ctr), tsis['center'])
    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break

