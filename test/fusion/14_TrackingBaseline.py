#!/usr/bin/env python2
from __future__ import print_function, division
import os
import sys
import json
import time

import cv2
import numpy as np

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..', '..'))

import geometry 
from util import *
from iterator import FrameIterator
from background import BackgroundModel, FrameDifference
import TSIUtil

"""
-------------------------------------------------------------------------------
Some Inititalization
-------------------------------------------------------------------------------
"""

VIEW = ("right", "center", "left")
with open ("/home/adib/My Git/traffic-monitoring/data/sync_25fps/common_point/result.json", "r") as f_buf : 
    GT = json.load (f_buf)

# cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)
img_path = '/home/adib/My Git/traffic-monitoring/data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/screen.png'

VP = VPLoader ()
ses_id = 1 
vp = VP.get_session (ses_id)
fi = FrameIteratorLoader.get_session (ses_id)

M = {} # matrix homography
M_inv = {} # matrix homography
imgs_color = {} # for saving image color each view
bm_dst = {}
fdiff_dst = {}
fdiff_view = {}
masks = {}
fgs = {}
prev_imgs_color = {}

for view in VIEW : 
    img = cv2.imread (img_path.format (ses_id, view), 1)

    points = GT['session{}'.format (ses_id)][view]
    corner = TSIUtil.get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points)

    # get rectangular homography mapping
    corner_gt = np.float32 (corner)
    corner_wrap = np.float32 ([[0,300],[0,0], [1000,0], [1000, 300]])
    M[view] = cv2.getPerspectiveTransform (corner_gt, corner_wrap)
    M_inv[view] = cv2.getPerspectiveTransform (corner_wrap, corner_gt)

    # for 3 frame difference
    prev_dst = [None, None]
    prev_view = [None, None]
    for i in range (2) : 
        img = next (fi[view])
        prev_imgs_color[view] = img
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

        dst = cv2.warpPerspective (img, M[view], (1000, 300))
        prev_dst[i] = dst
        prev_view[i] = img

    # bm_dst[view] = BackgroundModel (iter (prev_dst), detectShadows=False)
    # bm_dst[view].learn (tot_frame_init=2)
    fdiff_dst[view] = FrameDifference (*prev_dst)
    fdiff_view[view] = FrameDifference (*prev_view)

    mask_path = '../../data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/video_mask.png'.format (ses_id, view)
    masks[view] = cv2.imread (mask_path, 0)

"""
-------------------------------------------------------------------------------
Main Program
-------------------------------------------------------------------------------
"""
ctr = 0
while True:
    ctr += 1

    intersection = None
    disp = None
    t_begin = time.time ()

    for view in VIEW : 
        img_color = next (fi[view])
        imgs_color[view] = img_color
        img = cv2.cvtColor (img_color, cv2.COLOR_BGR2GRAY)

        dst = cv2.warpPerspective (img, M[view], (1000, 300))

        # fg = bm_dst[view].apply (dst)
        fg = fdiff_dst[view].apply (dst)
        fg = process_morphological (fg, iterations=2)

        fg_view = fdiff_view[view].apply (img)
        fg_view = process_morphological (fg_view, iterations=2)
        fg_view = cv2.bitwise_and (fg_view, masks[view]) # apply mask
        fgs[view] = fg_view

        if disp is None : 
            disp = dst
            intersection = fg 
        else : 
            disp = np.vstack ((disp, dst))
            intersection = cv2.bitwise_and (intersection, fg)


    blobs_PFM = get_contours (intersection) # get blobs 
    disp = {}
    for view in VIEW : 
        blobs_view = get_contours (fgs[view])
        disp[view] = prev_imgs_color[view]

        for b in blobs_PFM : 
            (x,y, w, h) = cv2.boundingRect (b)

            # get 4 corner represent this vehicle
            points = [
                    (x, y+h),
                    (x, y),
                    (x+w, y),
                    (x+w, y+h)
                ]

            # get its coordinate in this view
            bottom_box = TSIUtil.map_point (points, M_inv[view])

            # get its center
            center = TSIUtil.get_middle_point (bottom_box[0], bottom_box[2])

            # get corresponding blobs in view plane
            for bv in blobs_view : 
                if TSIUtil.is_contained (bv, center) : 
                    top_box = TSIUtil.get_top_polylines (bottom_box, bv)

                    # draw to view plane
                    disp[view] = TSIUtil.draw_3D_box (disp[view], bottom_box, top_box)
                    break

    for view in VIEW : 
        prev_imgs_color[view] = imgs_color[view]

    t_end = time.time ()
    print ('{} in {:.2F} s'.format (ctr, t_begin - t_end))

    # disp = np.vstack ((disp, intersection))
    disp = np.hstack (disp.values ())
    cv2.imshow ('default', disp)
    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break

