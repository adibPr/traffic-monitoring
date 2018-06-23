#!/usr/bin/env python2
from __future__ import print_function, division
import os
import sys
import json
import time
import random

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

cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)
img_path = '/home/adib/My Git/traffic-monitoring/data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/screen.png'

VP = VPLoader ()
ses_id = 5 
vp = VP.get_session (ses_id)
fi = FrameIteratorLoader.get_session (ses_id)

M = {} # matrix homography
M_inv = {} # matrix homography
imgs_color = {} # for saving image color each view
masks = {}
fgs = {}
prev_imgs_color = {}

# epi
epi_object = {}
fdiff_epi = {}

# tsi
tsi_object = {}
fdiff_tsi = {}

# miscelanous
empty = np.zeros ((1000, 700)).astype ('uint8')
random.seed (200)
unique_color = []
for i in range (100) : 
    unique_color.append ((random.randrange (0, 255), random.randrange (0, 255),random.randrange (0, 255)))

for view in VIEW : 
    img = cv2.imread (img_path.format (ses_id, view), 1)

    points = GT['session{}'.format (ses_id)][view]
    corner = TSIUtil.get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points)

    # get rectangular homography mapping
    corner_gt = np.float32 (corner)
    corner_wrap = np.float32 ([[0,300],[0,0], [1000,0], [1000, 300]])
    M[view] = cv2.getPerspectiveTransform (corner_gt, corner_wrap)
    M_inv[view] = cv2.getPerspectiveTransform (corner_wrap, corner_gt)

    # for epi object
    epi_object[view] = TSIUtil.EPI (M[view],
                size=(1000, 300),
                VDL_IDX=70, 
                VDL_SIZE=3, 
            )

    # for tsi object
    tsi_object[view] = TSIUtil.TSI (M[view])

    # for 3 frame difference
    prev_epi = [None, None]
    prev_tsi = [None, None]
    for i in range (2) : 
        img_color = img = next (fi[view])
        prev_imgs_color[view] = img
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

        dst = cv2.warpPerspective (img, M[view], (1000, 300))
        prev_epi[i] = epi_object[view].apply (dst)
        prev_tsi[i] = tsi_object[view].apply (dst)

    fdiff_epi[view] = FrameDifference (*prev_epi)
    fdiff_tsi[view] = FrameDifference (*prev_tsi)


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
    intersection_tsi = None
    disp = None
    t_begin = time.time ()

    for view in VIEW : 
        img_color = next (fi[view])
        imgs_color[view] = img_color
        img = cv2.cvtColor (img_color, cv2.COLOR_BGR2GRAY)

        dst = cv2.warpPerspective (img, M[view], (1000, 300))

        epi = epi_object[view].apply (dst)
        epi_fg = fdiff_epi[view].apply (epi)
        epi_fg = process_morphological (epi_fg, iterations=1)

        tsi = tsi_object[view].apply (dst)
        tsi_fg = fdiff_tsi[view].apply (tsi)
        tsi_fg = process_morphological (tsi_fg, iterations=2)

        if disp is None : 
            disp =  epi 
            intersection = epi_fg 
            intersection_tsi = tsi_fg
        else : 
            disp = np.vstack ((disp, epi))
            intersection = cv2.bitwise_and (intersection, epi_fg)
            intersection_tsi = cv2.bitwise_and (intersection_tsi, tsi_fg)

    t_end = time.time ()
    print ('{} in {:.2F} s'.format (ctr, t_begin - t_end))

    # draw line on dst
    dst = cv2.warpPerspective (imgs_color['center'], M['center'], (1000, 300))
    cv2.line (dst, (0, 70), (1000, 70), thickness=5, color=(255, 255, 0))

    pairs = []

    """ epi """
    intersection = intersection.transpose ()
    intersection = np.hstack ((intersection, empty))

    # get blobs
    blobs_epi = get_contours (intersection, min_area=0, min_width=2)
    # sort based on occurance
    blobs_epi = sorted (blobs_epi, key = lambda x : cv2.boundingRect(x)[0])
    # filter bsed on start index
    blobs_epi = [x for x in blobs_epi if cv2.boundingRect(x)[0] <= 10]

    # coloring
    intersection = cv2.cvtColor (intersection, cv2.COLOR_GRAY2BGR)
    # for b_idx, b in enumerate (blobs_epi) : 
    #     (x,y,w,h) = cv2.boundingRect (b)
    #     cv2.rectangle (intersection, (x,y), (x+w, y+h), unique_color[b_idx], 2)

    # intersection = draw_bounding_box_contours (intersection, blobs)
    # intersection = intersection[:10, :]

    """ tsi """
    # get blobs
    blobs_tsi = get_contours (intersection_tsi, min_area=0, min_width=2)
    # sorted based on occurance
    blobs_tsi = sorted (blobs_tsi, key = lambda x : cv2.boundingRect (x)[0])
    # filter based on line
    blobs_tsi = [x for x in blobs_tsi if cv2.boundingRect(x)[1] <= 75]

    # coloring
    intersection_tsi = cv2.cvtColor (intersection_tsi, cv2.COLOR_GRAY2BGR)
    for b_idx in range (min (len (blobs_epi), len (blobs_tsi))) : 
        # b = blobs_tsi[b_idx]
        # (x,y,w,h) = cv2.boundingRect (b)
        # cv2.rectangle (intersection_tsi, (x,y), (x+w, y+h), unique_color[b_idx], 2)

        pairs.append ((blobs_epi[b_idx], blobs_tsi[b_idx]))
        x,y,w,h = cv2.boundingRect (blobs_tsi[b_idx])
        real_y = y
        real_h = h
        x,y,w,h = cv2.boundingRect (blobs_epi[b_idx])
        real_x = h
        real_w = 20

        polylines = [(real_x, real_y+real_h), (real_x, real_y), (real_x+real_w, real_y), (real_x+real_w, real_y + real_h)]
        dst = TSIUtil.draw_polylines (dst, polylines)
        


    """ combine """
    # disp = cv2.cvtColor (disp, cv2.COLOR_GRAY2BGR)
    # disp = np.vstack ((dst, intersection_tsi, intersection))
    disp = dst
    cv2.imshow ('default', disp)

    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break

