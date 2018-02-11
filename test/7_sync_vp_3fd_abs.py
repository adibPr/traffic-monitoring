#!/usr/bin/env python

import os
import sys
import json

import cv2

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..'))


from iterator import FrameIterator
from background import BackgroundModel
from util import *
from geometry import get_extreme_tan_point, Line, get_extreme_tan_point_contours_real, get_extreme_side_point

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

session = {
    0 : {
        'center' : 569,
        'right' : 95,
        'left' : 420
    }
}

masks = {}
for _id in session : 
    masks[_id] = {}
    for view in session[_id] : 
        mask_path = '../data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/video_mask.png'.format (_id, view)
        masks[_id][view] = cv2.imread (mask_path, 0)

vps = {}
for _id in session : 
    vps[_id] = {}
    for view in session[_id] : 
        vp_path = '../data/gt/2016-ITS-BrnoCompSpeed/results/session{}_{}/system_dubska_bmvc14.json'.format (_id, view)

        with open (vp_path, 'r') as f_buff :
            """
            vp has 2 keys
            - cars, list of cars detected, its frame and posX and posY
            - camera_calibration, the calibration parameter result (pp, vp1, and vp2)
            """
            vp = json.load (f_buff)
            vps[_id][view] = {
                    'vp1' : vp['camera_calibration']['vp1'],
                    'vp2' : vp['camera_calibration']['vp2']
                }

"""
bms = {}
for _id in session: 
    bms[_id] = {} 
    for view in session[_id] : 
        fi = FrameIterator ('../data/sample/session{}_{}'.format (_id, view))
        bms[_id][view] = BackgroundModel (fi)
        print ("Learning for session {}-{}".format (_id, view))
        bms[_id][view].learn (tot_frame_init=1)
        print ("Done")
"""
prev_img = {}
fi = {}
for _id in session : 
    prev_img[_id] = {} 
    fi[_id] = {}

    for view in session[_id] : 
        fi[_id][view] = FrameIterator ('../data/sync/session{}_{}'.format (_id, view))
        prev_img[_id][view] = [next (fi[_id][view]), next (fi[_id][view])]


_id = 0
ctr = 0
while True :
    view_frame = None

    # load image from each view
    for view in session[_id] :
        img = next (fi[_id][view])

        vp1 = vps[_id][view]['vp1']
        vp2 = vps[_id][view]['vp2']

        # by background subtraction
        # remove shadows, i.e value 127
        # fg = cv2.threshold (fg, 200, 255, cv2.THRESH_BINARY)[1]

        # by three frame difference
        prev_diff = cv2.absdiff (prev_img[_id][view][-2], prev_img[_id][view][-1])
        curr_diff = cv2.absdiff (prev_img[_id][view][-2], img)
        # normal intersection
        fg = cv2.bitwise_and (prev_diff, curr_diff) 
        # dilation intersection
        prev_diff = cv2.dilate (prev_diff, kernel)
        curr_diff = cv2.dilate (curr_diff, kernel)
        fg2 = cv2.bitwise_and (prev_diff, curr_diff)
        # xor combination
        fg = cv2.bitwise_xor (fg, fg2)
        fg = cv2.cvtColor (fg, cv2.COLOR_BGR2GRAY) # convert into BG
        fg = cv2.threshold (fg, 10.0, 255.0, cv2.THRESH_BINARY)[1] # thresholding

        # remove noise
        fg = process_morphological (fg)
        # apply mask
        fg = cv2.bitwise_and (fg, masks[_id][view])
        # get blobs
        blobs = get_contours (fg)

        # drawing
        frame = cv2.cvtColor (fg, cv2.COLOR_GRAY2BGR)
        # frame = img
        frame = draw_bounding_box_contours (frame, blobs)

        # only consider that has object on it
        if blobs :

            for b in blobs : 
                extreme_point = get_extreme_side_point (vp1, vp2, b)
                for p in extreme_point : 
                    p = tuple ([int (_) for _ in p])

                    # draw it
                    frame = cv2.circle (frame, p, 10, (255,0,0), -1)

        # combine each view
        if view_frame is None :
            view_frame = frame
        else :
            view_frame = np.hstack ((frame, view_frame))

        prev_img[_id][view][0] = prev_img[_id][view][1]
        prev_img[_id][view][1] = img

    frame = view_frame
    # put text of iterator
    loc = (20, frame.shape[0]-20)
    cv2.putText (frame, 'Frame - {}'.format (ctr + 1), loc, cv2.FONT_HERSHEY_PLAIN, 3, (0, 128, 128), 4)

    # show image
    cv2.imshow ('default', frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break

    ctr += 1
