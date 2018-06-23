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
TOT_LANE = [3, 2, 2, 2, 2, 2, 3]  

VIEW = ("right", "center", "left")
with open ("/home/adib/My Git/traffic-monitoring/data/sync_25fps/common_point/result.json", "r") as f_buf : 
    GT = json.load (f_buf)

cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)
img_path = '/home/adib/My Git/traffic-monitoring/data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/screen.png'

VP = VPLoader ()
ses_id = 3 
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
empty = np.zeros ((300, 1000)).astype ('uint8')
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

flag_prev_object = False
list_height = []

"""
-------------------------------------------------------------------------------
Main Program
-------------------------------------------------------------------------------
"""
ctr = 0
while True:
    ctr += 1

    intersection_epi = None
    intersection_tsi = None
    disp = None
    t_begin = time.time ()

    for view in VIEW : 
        img_color = next (fi[view])
        imgs_color[view] = img_color
        img = cv2.cvtColor (img_color, cv2.COLOR_BGR2GRAY)

        dst = cv2.warpPerspective (img, M[view], (1000, 300))

        epi = epi_object[view].apply (img)
        epi_fg = fdiff_epi[view].apply (epi)
        epi_fg = process_morphological (epi_fg, iterations=2)

        tsi = tsi_object[view].apply (img)
        tsi_fg = fdiff_tsi[view].apply (tsi)
        tsi_fg = process_morphological (tsi_fg, iterations=2)

        if disp is None : 
            disp =  epi 
            intersection_epi = epi_fg 
            intersection_tsi = tsi_fg
        else : 
            disp = np.vstack ((disp, epi))
            intersection_epi = cv2.bitwise_and (intersection_epi, epi_fg)
            intersection_tsi = cv2.bitwise_and (intersection_tsi, tsi_fg)

    t_end = time.time ()
    # print ('{} in {:.2F} s'.format (ctr, t_begin - t_end))

    # drawing session
    dst_color = cv2.warpPerspective (imgs_color['center'], M['center'], (1000, 300))
    cv2.line (dst_color, (0, 71), (1000, 71), color=(255, 255, 0), thickness=2) 

    # """
    # making tsi and epi strip
    tsi_strip = intersection_tsi[:, 0:5]
    epi_strip = intersection_epi[0:3, :]
    empty[:, 0:5] = tsi_strip
    empty[70:73, :] =  epi_strip
    empty_color = cv2.cvtColor (empty, cv2.COLOR_GRAY2BGR)

    # get blobs for TSI
    tsi_blob = get_contours (tsi_strip, min_area=0, min_width=0)
    tmp = []
    for b in tsi_blob : 
        (x,y,w,h) = cv2.boundingRect (b)
        if y <= 71 <= y+h : 
            tmp.append ((y,h))
    tsi_blob = tmp
    assert len (tsi_blob) <= 1 
    if tsi_blob : 
        tsi_blob = tsi_blob[0]
        y,h = tsi_blob
        if flag_prev_object is False : 
            list_height.insert (0, tsi_blob)

        else : 
            # update width
            if list_height[0][1] < h : 
                list_height = list_height[1:]
                list_height.insert (0, tsi_blob)

        cv2.rectangle (empty_color, (0,y), (1, y+h), (0, 255, 0), 3)
        flag_prev_object = True
    else : 
        flag_prev_object = False

    # get blobs for EPI
    epi_blob = get_contours (epi_strip, min_area=-1, min_width=-1)
    epi_blob = [x for x in epi_blob if cv2.boundingRect (x)[2] > 10]
    epi_blob = sorted (epi_blob, key = lambda x : cv2.boundingRect(x)[0])
    for blob_idx, blob in enumerate (epi_blob) : 
        (x,y, w, h) = cv2.boundingRect (blob)
        # y += 70
        # cv2.rectangle (empty_color, (x,y), (x+w, y+h), (0, 255, 0), 3)
        if blob_idx >= len (list_height) : 
            print ("Overflow")
        else : 
            # draw into empty_color
            y, h = list_height[blob_idx]
            corner = [[x, y+h], [x, y], [x+w, y], [x+w, y+h]]
            dst_color = TSIUtil.draw_polylines (dst_color, corner)
    loc = (10, 30)
    cv2.putText (dst_color, 'Blob : {}'.format (len (epi_blob)), loc, cv2.FONT_HERSHEY_PLAIN, 3, (0, 128, 128), 4)
    # """


    intersection_tsi = cv2.cvtColor (intersection_tsi, cv2.COLOR_GRAY2BGR)
    intersection_epi = cv2.cvtColor (intersection_epi, cv2.COLOR_GRAY2BGR)
    """ combine """
    disp = np.vstack ((dst_color, empty_color, intersection_tsi, intersection_epi))
    cv2.imshow ('default', disp)

    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break

