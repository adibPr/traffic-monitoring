#!/usr/bin/env python2
from __future__ import print_function, division
import os
import sys
import json

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
ses_id = 0 
vp = VP.get_session (ses_id)
fi = FrameIteratorLoader.get_session (ses_id)

M = {} # matrix homography
imgs_color = {} # for saving image color each view
fgs = {}
fdiff_view = {} # frame difference camera image
masks = {}

for view in VIEW : 
    img = cv2.imread (img_path.format (ses_id, view), 1)

    points = GT['session{}'.format (ses_id)][view]
    corner = TSIUtil.get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points)

    # get rectangular homography mapping
    corner_gt = np.float32 (corner)
    corner_wrap = np.float32 ([[0,300],[0,0], [1000,0], [1000, 300]])
    M[view] = cv2.getPerspectiveTransform (corner_gt, corner_wrap)

    # for 3 frame difference
    prev_img = [None, None]
    for i in range (2) : 
        img = next (fi[view])
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

        # save background
        prev_img[i] = img 

    fdiff_view[view] = FrameDifference (*prev_img)

    mask_path = '../../data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/video_mask.png'.format (ses_id, view)
    masks[view] = cv2.imread (mask_path, 0)

ss = list (prev_img[0].shape)
ss.append (int (3))
ss=tuple (ss)
print (ss)
print (type (prev_img[0].shape))

fourcc = cv2.VideoWriter_fourcc (*'XVID')
video = cv2.VideoWriter ('video.avi', fourcc, 25.0, prev_img[0].shape)

"""
-------------------------------------------------------------------------------
Main Program
-------------------------------------------------------------------------------
"""
# what I want is a function to generate 3D using Vanishing Point (it already exist)
ctr = 0
while True :
    ctr += 1
    frame = None

    for view in VIEW : 
        # load image
        imgs_color[view] = img_color = next (fi[view])
        img = cv2.cvtColor (img_color, cv2.COLOR_BGR2GRAY)
        
        # extract object by 3 frame difference
        fgs[view] = fdiff_view[view].apply (img)

        # apply morphological operation
        fgs[view] = process_morphological (fgs[view])

        # apply mask
        fgs[view] = cv2.bitwise_and (fgs[view], masks[view])

        # extract blobs
        blobs = get_contours (fgs[view])

        # draw 3D bounding box
        for bl in blobs : 
            img_color = geometry.draw_bounding_box_contours_3D (
                    img_color, 
                    bl, 
                    vp[view]['vp1'], 
                    vp[view]['vp2']
                )
        if frame is None : 
            frame = img_color 
        else :
            frame = np.hstack ((frame, img_color))

    cv2.imshow ('default', frame)
    print (img_color.shape)
    video.write (img_color)
    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        video.release ()
        break

