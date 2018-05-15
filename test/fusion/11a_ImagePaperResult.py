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
imgs_color = {} # for saving image color each view
dsts_color = {}
dsts = {}
fgs = {}
fdiff_tsi = {} # frame difference tsi image
fdiff_view = {} # frame difference camera image
tsi_object  = {}
posision_x_gt = [980, 758, 711]
posision_x_gt = posision_x_gt[::-1]

for view in VIEW : 
    img = cv2.imread (img_path.format (ses_id, view), 1)
    
    # fi[view] = FrameIterator ('/home/adib/My Git/traffic-monitoring/test/fusion/result/sync_25fps_paper/{}'.format (view)) 

    points = GT['session{}'.format (ses_id)][view]
    corner = TSIUtil.get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points)

    # get rectangular homography mapping
    corner_gt = np.float32 (corner)
    corner_wrap = np.float32 ([[0,300],[0,0], [1000,0], [1000, 300]])
    M[view] = cv2.getPerspectiveTransform (corner_gt, corner_wrap)

    # initialize tsi object
    tsi_object[view] = TSIUtil.TSI (M[view], VDL_IDX=100)

    # for 3 frame difference
    prev_img = [None, None]
    prev_tsi = [None, None]
    for i in range (2) : 
        img = next (fi[view])
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

        # save background
        prev_img[i] = img 

        # save tsi
        prev_tsi[i] = tsi_object[view].apply (img)

    fdiff_view[view] = FrameDifference (*prev_img)
    fdiff_tsi[view] = FrameDifference (*prev_tsi)

    fi[view].skip (50)


ctr = 50 
while True:
    ctr += 1
    print (ctr)
    disp_tsi = None
    intersection = None

    for view in VIEW : 
        img_color = next (fi[view])
        imgs_color[view] = img_color
        img = cv2.cvtColor (img_color, cv2.COLOR_BGR2GRAY)

        dst = cv2.warpPerspective (img, M[view], (1000, 300))
        dsts[view] = dst

        tsi = tsi_object[view].apply (img)
        tsi_fg = fdiff_tsi[view].apply (tsi)
        
        fg = fdiff_view[view].apply (img, iterations=2)
        fgs[view] = fg

        if disp_tsi is None : 
            disp_tsi = tsi
            intersection = tsi_fg
        else : 
            disp_tsi = np.vstack ((tsi, disp_tsi))
            intersection = cv2.bitwise_and (intersection, tsi_fg)

        # cv2.imwrite ('result/sync_25fps_paper/{}/{}.jpg'.format (view, str (ctr).zfill (4)), img_color)

    blobs = TSIUtil.get_contours (intersection)
    blobs = TSIUtil.get_most_left_blobs (blobs, n=3)
            
    intersection = cv2.cvtColor (intersection, cv2.COLOR_GRAY2BGR)
    intersection = draw_bounding_box_contours (intersection, blobs)
    disp_tsi = cv2.cvtColor (disp_tsi, cv2.COLOR_GRAY2BGR)
    dst = cv2.cvtColor (dst, cv2.COLOR_GRAY2BGR)
    disp = np.vstack ((disp_tsi, intersection, dst))

    cv2.imshow ('default', disp)
    if ctr ==  101 : 
        for view in VIEW : 
            dst = cv2.cvtColor (dsts[view], cv2.COLOR_GRAY2BGR)
            points = GT['session{}'.format (ses_id)][view]
            corner = np.float32 (TSIUtil.get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points))
            M_inv = cv2.getPerspectiveTransform (corner_wrap, corner)

            img_color = imgs_color[view]
            # img_color = fgs[view]
            # img_color = cv2.cvtColor (img_color, cv2.COLOR_GRAY2BGR)

            view_blob = get_contours (fgs[view])

            for b_idx, b in enumerate (blobs) : 
                x,y,w,h = tsi_object[view].get_approximate_length (b, padding=0)
                x = posision_x_gt[b_idx]-w

                points = [
                        [x, y+h],
                        [x, y],
                        [x+w, y],
                        [x+w, y+h]
                    ]

                # bottom polylines
                bottom_polylines = TSIUtil.map_point (points, M_inv)

                dst = TSIUtil.draw_polylines (dst, points, color=(255, b_idx * 255, 255), thickness=3)

                for vb in view_blob : 
                    if TSIUtil.is_contained (vb, TSIUtil.get_middle_point (bottom_polylines[1], bottom_polylines[3])) : 
                        # top polylines
                        top_polylines = TSIUtil.get_top_polylines (bottom_polylines, vb)

                        img_color = TSIUtil.draw_3D_box (img_color, bottom_polylines, top_polylines, color=(255, b_idx * 255, 255), thickness=3)
                        break


            cv2.imwrite ('result/view.{}.{}.jpg'.format (view, ctr), img_color)
            cv2.imwrite ('result/fg.{}.{}.jpg'.format (view, ctr), fgs[view])
            cv2.imwrite ('result/dst.{}.{}.jpg'.format (view, ctr), dst)
        cv2.imwrite ('result/disp_tsi.{}.jpg'.format (ctr), disp)
        break

    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break

