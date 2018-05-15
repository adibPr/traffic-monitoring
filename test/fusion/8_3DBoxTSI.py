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
from background import BackgroundModel

def draw_polylines (img, corner, color=(0,0,255), thickness=5) : 
    img = img.copy ()

    cornerInt = []
    for c in corner : 
        cornerInt.append (tuple ([int (_) for _ in c]))

    corner = np.array (corner, np.int32)
    corner = corner.reshape ((-1, 1, 2))

    cv2.polylines (img, [corner], True , color, thickness)
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

cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)
img_path = '/home/adib/My Git/traffic-monitoring/data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/screen.png'

ses_id = 5 
vp = VP.get_session (ses_id)
fi = FrameIteratorLoader.get_session (ses_id)

M = {} # matrix homography
ts_img = {} # initalization of time spatial image
prev_img = {} # prev 2 image for 3 frame differance
prev_tsi = {} # prev 2 image for 3 frame difference time spatial image
bms = {} # background model MoG
VDL_IDX = 100 # column index of VDL
VDL_SIZE = 5  # size of VDL
masks = {} # for masks 
color = (0,0,255)
imgs_color = {} # for saving image color each view
fgs = {}

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

    # for 3 frame difference
    prev_img[view] = [None, None]
    for i in range (2) : 
        img = next (fi[view])
        img_color = img.copy ()
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

        # save background
        prev_img[view][i] = img 

    prev_tsi[view] = [None, None]

    # laod mask
    mask_path = '../../data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/video_mask.png'.format (ses_id, view)
    masks[view] = cv2.imread (mask_path, 0)

ctr = 0
while True:
    ctr += 1

    for view in VIEW : 
        img_color = next (fi[view])
        imgs_color[view] = img_color
        img = cv2.cvtColor (img_color, cv2.COLOR_BGR2GRAY)

        # normal 3FD
        prev_intersect = cv2.threshold (cv2.absdiff (prev_img[view][1], prev_img[view][0]), 25, 255, cv2.THRESH_BINARY)[1]
        next_intersect = cv2.threshold (cv2.absdiff (img, prev_img[view][1]), 25, 255, cv2.THRESH_BINARY)[1]

        # litle  3FD enhanchement
        P1 = cv2.bitwise_and (prev_intersect, next_intersect)
        prev_intersect_dilate = process_morphological (prev_intersect) 
        next_intersect_dilate = process_morphological (next_intersect)
        P2 = cv2.bitwise_and (prev_intersect_dilate, next_intersect_dilate)

        fg = cv2.bitwise_and (P2, masks[view])
        fg = process_morphological (fg)

        fgs[view] = fg

        # update
        prev_img[view][0] = prev_img[view][1]
        prev_img[view][1] = img

    print (ctr)
    if ctr == 284 :  
        with open ("result/3_3D_box/box_ground.json", 'r') as f_b : 
            boxes_ground = json.load (f_b)

        for view in VIEW : 
            # first, draw ground truth
            points = GT['session{}'.format (ses_id)][view]
            corner = np.float32 (get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points))

            """
            # get dst_color with rectangle
            box_ground = np.matrix ([
                [VDL_IDX + 600, selected[1] + selected[3], 1], # (bottom, left)
                [VDL_IDX + 600, selected[1], 1], # (top, left)
                [VDL_IDX + 600 + selected[2], selected[1], 1], # (top, right)
                [VDL_IDX + 600 + selected[2], selected[1] + selected[3], 1] # (bottom, right)
            ]).transpose ()
            # """

            # get inverse perspective
            M_inv = cv2.getPerspectiveTransform (corner_wrap, corner)
            
            """
            # draw on the real image
            # box_ground = M_inv.dot (box_ground).transpose () # get inverse location of box
            # box_ground /= box_ground[:, 2] # divide by homogoneous scale
            # box_ground = box_ground[:, :-1].astype ('int').tolist () # convert into index
            # """

            box_ground = np.matrix (boxes_ground[view])
                

            # img_color = cv2.cvtColor (fg, cv2.COLOR_GRAY2BGR)
            inverse_img = draw_polylines (imgs_color[view], box_ground.tolist (), color=(0,255,0), thickness=2)

            # get blobs
            blobs = get_contours (fgs[view])

            # get the blobs that within box_ground
            is_found = False
            for b in blobs : 
                (x,y, w, h) = cv2.boundingRect (b)
                for cp in box_ground.tolist () : 
                    if cp[0] >= x and cp[0] <= x+w and cp[1] >= y and cp[1] <= y+h : 
                        is_found = True
                        break

                if is_found : 
                    # then construct line vertical height
                    max_cp_y = np.min (box_ground[:, 1]) - 3 # small tradeoff
                    box_floor = []
                    for cp in box_ground.tolist () : 
                        cp_h = list (cp)
                        cp_h[1] -= abs (y - max_cp_y)

                        cv2.line (inverse_img, tuple (cp), tuple (cp_h), (0, 255, 0), 2) 

                        box_floor.append (cp_h)

                    inverse_img = draw_polylines (inverse_img, box_floor, color=(0,255,0), thickness=2)
                    break


            cv2.imwrite ('result/{}-{}.jpg'.format (view, ctr), inverse_img)

        break
    # show image
    cv2.imshow ('default', img_color)

    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break

