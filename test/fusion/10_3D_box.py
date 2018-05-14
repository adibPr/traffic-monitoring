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

def get_middle_point (p1, p2) : 
    x1 = int ((p1[0] + p2[0]) / 2)
    x2 = int ((p1[1] + p2[1]) / 2)
    return (x1, x2)

def is_contained (blob, middle_point) : 
    (x,y, w, h) = cv2.boundingRect (blob)
    if middle_point[0] >= x and middle_point[0] <= x + w \
            and middle_point[1] >= y and middle_point[1] <=  y + h :
        return True
    return False

VP = VPLoader ()

VIEW = ("right", "center", "left")
with open ("/home/adib/My Git/traffic-monitoring/data/sync_25fps/common_point/result.json", "r") as f_buf : 
    GT = json.load (f_buf)

# cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)
img_path = '/home/adib/My Git/traffic-monitoring/data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/screen.png'

ses_id = 5 
vp = VP.get_session (ses_id)
fi = FrameIteratorLoader.get_session (ses_id)

M = {} # matrix homography
prev_img = {} # prev 2 image for 3 frame differance
bms = {} # background model MoG
imgs_color = {} # for saving image color each view
fgs = {}
masks = {}
HEIGHT = 400 # constant height

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

    # for 3 frame difference
    prev_img[view] = [None, None]
    for i in range (2) : 
        img = next (fi[view])
        img_color = img.copy ()
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

        # save background
        prev_img[view][i] = None

    mask_path = '../../data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/video_mask.png'.format (ses_id, view)
    masks[view] = cv2.imread (mask_path, 0)

ctr = 0
while True:
    ctr += 1

    frame = None
    frame_PFM = None
    intersection = None # PFM TSI

    for view in VIEW : 
        img_color = next (fi[view])
        imgs_color[view] = img_color
        img = cv2.cvtColor (img_color, cv2.COLOR_BGR2GRAY)

        # """
        # background subtraction by 3 frame difference
        if prev_img[view][0] is None : 
            prev_img[view][0] = img
        elif prev_img[view][1] is None : 
            prev_img[view][1] = img 
        else : 
            prev_intersect = cv2.threshold (cv2.absdiff (prev_img[view][1], prev_img[view][0]), 25, 255, cv2.THRESH_BINARY)[1]
            next_intersect = cv2.threshold (cv2.absdiff (img, prev_img[view][1]), 25, 255, cv2.THRESH_BINARY)[1]

            # by 3 frame of "Vehicle speed measurement based on gray constraint optical flow algorithm"
            P1 = cv2.bitwise_and (prev_intersect, next_intersect)
            prev_intersect_dilate = process_morphological (prev_intersect) 
            next_intersect_dilate = process_morphological (next_intersect)
            P2 = cv2.bitwise_and (prev_intersect_dilate, next_intersect_dilate)

            fg = cv2.bitwise_and (P2, masks[view])
            fg = process_morphological (fg)

            fgs[view] = fg

            # """

            prev_img[view][0] = prev_img[view][1]
            prev_img[view][1] = img 

    print (ctr)
    if ctr == 387 :
        with open ('box_ground.json', 'r') as f_b : 
            bbox = json.load (f_b)

        for view in VIEW : 
            # first, draw ground truth
            points = GT['session{}'.format (ses_id)][view]
            corner = np.float32 (get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points))

            M_inv = cv2.getPerspectiveTransform (corner_wrap, corner)

            blobs = get_contours (fgs[view])

            # convert foreground to color so can be enhanched
            fgs[view] = imgs_color[view] # 
            # fgs[view] = cv2.cvtColor (fgs[view], cv2.COLOR_GRAY2BGR)

            for b_idx, b in enumerate (bbox[view]) : 

                la = Line.from_two_points (b[0], b[2])
                lb = Line.from_two_points (b[1], b[3])

                middle_point = la.get_intersection (lb)
                middle_point = tuple ([int (_) for _ in middle_point])
                color = (255, 255, b_idx * 255)
                thickness = 2

                fgs[view] = draw_polylines (fgs[view], b, color, thickness )

                b = np.matrix (b)

                # get blobs that contain this 
                is_found = False
                for bl in blobs : 
                    if is_contained (bl, middle_point) :
                        print ("Found middle")
                        is_found = True
                        max_cp_y = np.max (b[:, 1]) - 3 # small tradeoff
                        box_floor = []
                        (x,y,w,h) = cv2.boundingRect (bl)

                        for cp in b.tolist () : 
                            cp_h = list (cp)
                            # cp_h[1] -= abs (y - max_cp_y)
                            cp_h[1] -= HEIGHT

                            cv2.line (fgs[view], tuple (cp), tuple (cp_h), color, thickness=thickness) 

                            box_floor.append (cp_h)

                        fgs[view] = draw_polylines (fgs[view], box_floor, color, thickness)
                        break


            cv2.imwrite ('result/{}-{}.jpg'.format (view, ctr), fgs[view])
        break

    # cv2.imshow ('default', frame)
    # if (cv2.waitKey(1) & 0xFF == ord('q')) :
    #     break
