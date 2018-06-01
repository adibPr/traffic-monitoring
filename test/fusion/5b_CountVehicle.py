#!/usr/bin/env python3
from __future__ import print_function, division
import os
import sys
import json

import cv2
import numpy as np

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..', '..'))

from geometry import get_extreme_tan_point, Line, get_extreme_side_point, find_right_most_point
from util import *
from iterator import FrameIterator
from background import  *
import TSIUtil

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

def TSI_get_contours (frame_binary, min_area=200, min_width=50) :
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

# cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)
img_path = '/home/adib/My Git/traffic-monitoring/data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/screen.png'
ses_id = 0 

result = []
for ses_id in range (7) :
    print ("Analyzing ses : {}".format (ses_id))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('result/TSI_video/TSI/PFI_ses_{}.avi'.format (ses_id),fourcc, 25.0, (1000, 1230))

    vp = VP.get_session (ses_id)
    fi = FrameIteratorLoader.get_session (ses_id)

    M = {} # matrix homography
    ts_img = {} # initalization of time spatial image
    tsi_object  = {}
    fdiff_tsi = {} # frame difference tsi image
    fdiff_view = {}
    masks = {}

    for view in VIEW : 
        img = cv2.imread (img_path.format (ses_id, view), 1)

        points = GT['session{}'.format (ses_id)][view]
        corner = get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points)

        # get rectangular homography mapping
        corner_gt = np.float32 (corner)
        corner_wrap = np.float32 ([[0,300],[0,0], [1000,0], [1000, 300]])
        M[view] = cv2.getPerspectiveTransform (corner_gt, corner_wrap)

        # for initialization
        ts_img[view] = None # np.zeros ((300, 1000, 3)) 
        tsi_object[view] = TSIUtil.TSI (M[view], VDL_IDX=0)

        # for 3 frame difference
        prev_img = [None, None]
        prev_tsi = [None, None]

        for i in range (2) : 
            img = next (fi[view])
            img_color = img.copy ()
            img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

            dst = cv2.warpPerspective (img, M[view], (1000, 300))

            # save background
            prev_img[i] = dst
            prev_tsi[i] = tsi_object[view].apply (img)

        fdiff_tsi[view] = FrameDifference (*prev_tsi)
        fdiff_view[view] = FrameDifference (*prev_img)

        mask_path = '../../data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/video_mask.png'.format (ses_id, view)
        masks[view] = cv2.imread (mask_path, 0)

    center_list = [] # for tracking
    T_DIST = 1 # threshold distance
    T_XRANGE = 20 # threshold of consideration of out vehicle
    tot_vehicle = 0
    ctr = 0

    while True:
        try : 
            ctr += 1

            frame = None
            disp_tsi = None
            intersection = None 

            for view in VIEW : 
                img_color = next (fi[view])
                img = cv2.cvtColor (img_color, cv2.COLOR_BGR2GRAY)

                # for displyaing background result
                # dst = cv2.warpPerspective (P2, M[view], (1000,300))
                
                dst = cv2.warpPerspective (img, M[view], (1000, 300))

                # apply tsi
                # tsi = tsi_object[view].apply (img)
                # tsi_fg = fdiff_tsi[view].apply (tsi, iterations=2)

                view_fg = fdiff_view[view].apply (dst, iterations=2)

                # time spatial construction
                if disp_tsi is None : 
                    disp_tsi =  dst
                    intersection = view_fg 
                else : 
                    disp_tsi = np.vstack ((dst, disp_tsi))
                    intersection = cv2.bitwise_and (intersection, view_fg)


            frame = disp_tsi

            # add morphological process to intersection
            intersection = process_morphological (intersection)

            # i want draw bounding box over the intersection only
            # blobs = TSI_get_contours (intersection) # get blobs
            # intersection = draw_bounding_box_contours (intersection, blobs) # draw bounding box
            blobs = get_contours (intersection)
            intersection = cv2.cvtColor (intersection, cv2.COLOR_GRAY2BGR) # convert intersection into BGR

            # get bounding box that touch begin line
            for b in blobs : 
                (x,y, w, h) = cv2.boundingRect (b)
                center_y = (y + h) / 2
                center_x = (x + w) / 2
                if abs (center_x - 200) <= 3 : 
                    cv2.rectangle (intersection, (x,y), (x+w, y+h), (0, 0, 255), 2)
                    tot_vehicle += 1
                else : 
                    cv2.rectangle (intersection, (x,y), (x+w, y+h), (0, 255, 0), 2)

                        
            # convert frame to color, so we can vstack with intersection
            frame = cv2.cvtColor (frame, cv2.COLOR_GRAY2BGR)

            # concat view
            frame = np.vstack ((frame, intersection))
            pad = np.zeros ((30, frame.shape[1], len (frame.shape))).astype ('uint8')
            frame = np.vstack ((
                frame, 
                pad
            ))

            # add counter
            loc = (10, frame.shape[0] - 5)
            cv2.putText (frame, 'Vehicle - {}'.format (tot_vehicle), loc, cv2.FONT_HERSHEY_PLAIN, 3, (0, 128, 128), 4)
            out.write (frame)

            # cv2.imshow ('default', frame)
            # if (cv2.waitKey(1) & 0xFF == ord('q')) :
            #     break

        except (KeyboardInterrupt,StopIteration) as e : 
            # add the leftover tot_vehicle
            for b in blobs : 
                (x,y,w,h) = cv2.boundingRect (b)
                center_x = (x+w) / 2
                if center_x < 500 : 
                    tot_vehicle += 1

            print ("Tot Vehicle : {}".format (tot_vehicle))
            result.append (tot_vehicle)
            out.release ()

            break
with open ('result/TSI_video/PFI_result.json', 'w') as f_buff : 
    json.dump (result, f_buff)
