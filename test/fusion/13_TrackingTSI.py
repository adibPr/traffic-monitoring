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
ses_id = 4 
vp = VP.get_session (ses_id)
fi = FrameIteratorLoader.get_session (ses_id)

M = {} # matrix homography
imgs_color = {} # for saving image color each view
fgs = {}
dsts = {}
prev_dsts = {}
fdiff_view = {} # frame difference camera image
fdiff_tsi = {} # frame difference tsi image
fdiff_epi = {} # frame difference epi image
bm_epi = {}
tsi_object  = {}
epi_object = {}
masks = {}

for view in VIEW : 
    img = cv2.imread (img_path.format (ses_id, view), 1)

    points = GT['session{}'.format (ses_id)][view]
    corner = TSIUtil.get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points)

    # get rectangular homography mapping
    corner_gt = np.float32 (corner)
    corner_wrap = np.float32 ([[0,300],[0,0], [1000,0], [1000, 300]])
    M[view] = cv2.getPerspectiveTransform (corner_gt, corner_wrap)

    # initialize tsi object
    tsi_object[view] = TSIUtil.TSI (M[view], VDL_IDX=0)
    # epi_object[view] = TSIUtil.EPI (M[view], VDL_IDX=75)

    # for 3 frame difference
    prev_img = [None, None]
    prev_tsi = [None, None]
    prev_epi = [None, None]
    for i in range (2) : 
        img = next (fi[view])
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

        dst = cv2.warpPerspective (img, M[view], (1000, 300))

        # save background
        prev_img[i] = img 

        # save tsi
        prev_tsi[i] = tsi_object[view].apply (img)
        prev_epi[i] = dst[70:80:, :]

    # fdiff_view[view] = FrameDifference (*prev_img)
    fdiff_tsi[view] = FrameDifference (*prev_tsi)
    fdiff_epi[view] = FrameDifference (*prev_epi)
    bm_epi[view] = BackgroundModel (iter (prev_epi))
    bm_epi[view].learn (tot_frame_init=2)

    mask_path = '../../data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/video_mask.png'.format (ses_id, view)
    masks[view] = cv2.imread (mask_path, 0)

"""
-------------------------------------------------------------------------------
Main Program
-------------------------------------------------------------------------------
"""
# what I want is a function to generate 3D using Vanishing Point (it already exist)
ctr = 0
while True:
    ctr += 1

    disp_tsi = None
    intersection = None
    disp_epi = None
    t_begin = time.time ()

    for view in VIEW : 
        img_color = next (fi[view])
        imgs_color[view] = img_color
        img = cv2.cvtColor (img_color, cv2.COLOR_BGR2GRAY)

        dst = cv2.warpPerspective (img, M[view], (1000, 300))

        # epi
        epi = dst[70:80,:]
        epi_fg = bm_epi[view].apply (epi) 
        # epi_fg = fdiff_epi[view].apply (epi) 

        # tsi
        # tsi = tsi_object[view].apply (img)
        # tsi_fg = fdiff_tsi[view].apply (tsi, iterations=2)

        tsi = dst

        if disp_tsi is None : 
            disp_tsi = tsi
            disp_epi = epi_fg
            intersection = epi_fg
        else : 
            disp_tsi = np.vstack ((tsi, disp_tsi))
            disp_epi = np.vstack ((epi_fg, disp_epi))
            intersection = cv2.bitwise_and (intersection, epi_fg)

    t_end = time.time ()
    print ('{} in {:.2F} s'.format (ctr, t_begin - t_end))

    disp_tsi = np.vstack ((disp_tsi, intersection))
    cv2.imshow ('default', disp_tsi)
    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break

