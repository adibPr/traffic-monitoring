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
VDL_IDX = 1 # column index of VDL
VDL_SIZE = 5  # size of VDL
VDL_SCALE = 3/4 # scale of VDL
imgs_color = {} # for saving image color each view
dsts_color = {}

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
        prev_img[view][i] = None

    prev_tsi[view] = [None, None]

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

        # warp perspective
        dst = cv2.warpPerspective (img, M[view], (1000, 300))
        dst_color = cv2.warpPerspective (img_color, M[view], (1000, 300))
        dsts_color[view] =  dst_color

        # """
        # background subtraction by 3 frame difference
        if prev_img[view][0] is None : 
            prev_img[view][0] = dst
        elif prev_img[view][1] is None : 
            prev_img[view][1] = dst
        else : 
            prev_intersect = cv2.threshold (cv2.absdiff (prev_img[view][1], prev_img[view][0]), 25, 255, cv2.THRESH_BINARY)[1]
            next_intersect = cv2.threshold (cv2.absdiff (dst, prev_img[view][1]), 25, 255, cv2.THRESH_BINARY)[1]

            # by 3 frame of "Vehicle speed measurement based on gray constraint optical flow algorithm"
            P1 = cv2.bitwise_and (prev_intersect, next_intersect)
            prev_intersect_dilate = process_morphological (prev_intersect) 
            next_intersect_dilate = process_morphological (next_intersect)
            
            frame_PFM = cv2.bitwise_and (prev_intersect_dilate, next_intersect_dilate)
            frame_PFM = cv2.cvtColor (frame_PFM, cv2.COLOR_GRAY2BGR)
            # """

            prev_img[view][0] = prev_img[view][1]
            prev_img[view][1] = dst

        # take only the LINE_COL column
        border_img = dst[:, VDL_IDX:VDL_IDX+VDL_SIZE].copy ()

        # display VDL
        cv2.line (dst_color, (VDL_IDX, 0), (VDL_IDX, dst.shape[1]), color=(0, 255, 200), thickness=10)

        # time spatial construction
        if ts_img[view] is None : 
            ts_img[view] = border_img
        else : 
            ts_img[view] = np.hstack ((border_img, ts_img[view]))

        # PFM TSI
        cur_disp = ts_img[view]
        if ts_img[view].shape[1] > 1000 : 
            cur_disp = ts_img[view] = ts_img[view][:, :-VDL_SIZE]

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

        if intersection is None : 
            intersection = cur_disp

        else : 
            intersection = cv2.bitwise_and (intersection, cur_disp)


    print (ctr)
    # start when everything is done
    if intersection.shape[1] != 1000 : 
        continue

    intersection = process_morphological (intersection) # add morphological process
    blobs = TSI_get_contours (intersection) # get blobs
    intersection = cv2.cvtColor (intersection, cv2.COLOR_GRAY2BGR) # convert intersection into BGR
    intersection = draw_bounding_box_contours (intersection, blobs) # draw bounding box

    frame = np.vstack ((dsts_color['left'], frame_PFM, intersection))

    loc = (5, 30)
    cv2.putText (frame, 'Frame - {}'.format (ctr), loc, cv2.FONT_HERSHEY_PLAIN, 3, (255, 200, 128), 2)


    # get rectangular area based on VDL width
    prev_x = 1000
    rect_params = [list (cv2.boundingRect (b)) for b in blobs] # get all bounding box parameter
    rect_params = sorted (rect_params, key = lambda p : p[0])[:2] # sorted based on the x value
    # got maximum x+2
    max_x = max ([(VDL_SCALE * VDL_SIZE) * (r[0]+ r[2]) for r in rect_params])
    min_x = min ([(VDL_SCALE * VDL_SIZE) * r[0] for r in rect_params])
    diff = max_x - min_x # rescale

    for r_idx, r in enumerate (rect_params) : 
        # (x,y, w, h)
        rect_params[r_idx][2] = int ((VDL_SCALE) * VDL_SIZE * r[2]) #convert to actual length
        rect_params[r_idx][0] -= rect_params[0][0]  # rescale position of x
        # rect_params[r_idx][0] += (1000 - max_x)  # rescale position of x
        rect_params[r_idx][0] = int (rect_params[r_idx][0] * VDL_SCALE * VDL_SIZE) # convert x to actual length

        (x,y,w,h) = rect_params[r_idx]

        # draw both on frame
        cv2.rectangle (
                frame, 
                (VDL_IDX + 1000 - int (diff) + x, y), 
                (VDL_IDX + 1000 - int (diff) + x + w, y + h),
                color=(255, 0, 255 * (r_idx % 2)), 
                thickness=5
            )

    if ctr == 387 :
        scale_x = VDL_IDX + 1000 - int (diff)
        bbox = {}
        for view in VIEW : 
            # first, draw ground truth
            points = GT['session{}'.format (ses_id)][view]
            corner = np.float32 (get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points))
            # img_color = draw_polylines (imgs_color[view], corner)

            bbox[view] = []
            img_color = imgs_color[view]

            for (x,y,w,h) in rect_params : 

                box_ground = np.matrix ([
                    [scale_x + x, y + h, 1], # (bottom, left)
                    [scale_x + x, y, 1], # (top, left)
                    [scale_x + x + w, y, 1], # (top, right)
                    [scale_x + x + w, y + h, 1] # (bottom, right)
                ]).transpose ()

                # get inverse perspective
                M_inv = cv2.getPerspectiveTransform (corner_wrap, corner)
                
                # draw on the real image
                box_ground = M_inv.dot (box_ground).transpose () # get inverse location of box
                box_ground /= box_ground[:, 2] # divide by homogoneous scale
                box_ground = box_ground[:, :-1].astype ('int').tolist () # convert into index
                bbox[view].append (box_ground)
                
                # draw on the real image
                inverse_dst_color = cv2.warpPerspective (dsts_color[view], M_inv, (img_color.shape[1], img_color.shape[0]) )
                # inverse_img = cv2.add (img_color, inverse_dst_color)
                img_color = draw_polylines (img_color, bbox[view][-1])

            cv2.imwrite ('result/{}-{}.jpg'.format (view, ctr), img_color)

        with open ('box_ground.json', 'w') as f_b : 
            json.dump (bbox, f_b)

        break

    cv2.imshow ('default', frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break
