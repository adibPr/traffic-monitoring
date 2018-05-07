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
from geometry import get_extreme_tan_point, Line, get_extreme_tan_point_contours

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

bms = {}
for _id in session: 
    bms[_id] = {} 
    for view in session[_id] : 
        fi = FrameIterator ('../data/sample/session{}_{}'.format (_id, view))
        bms[_id][view] = BackgroundModel (fi)
        print ("Learning for session {}-{}".format (_id, view))
        bms[_id][view].learn (tot_frame_init=1)
        print ("Done")

_id = 0
ctr = 0
while True :
    view_frame = None

    # load image from each view
    for view in session[_id] :
        fpath = '../data/sample/session0_{}/{:04d}.jpg'.format (
                view,
                session[_id][view] + ctr
            )

        img = cv2.imread (fpath, 1)
        frame = img
        fg = bms[_id][view].apply (img)

        # remove shadows, i.e value 127
        fg = cv2.threshold (fg, 200, 255, cv2.THRESH_BINARY)[1]
        # remove noise
        fg = process_morphological (fg)
        # apply mask
        fg = cv2.bitwise_and (fg, masks[_id][view])
        # get blobs
        blobs = get_contours (fg)

        # only consider that has object on it
        if blobs :

            for vp in (vps[_id][view].values ()) :
                # select first blobs
                this_blobs = blobs[0]
                # get its tan point
                tan_left, tan_right = get_extreme_tan_point_contours (vp, blobs)

                # construct line from tangent point to vp2
                l_left_vp = Line.from_two_points (tan_left, vp)
                l_right_vp = Line.from_two_points (tan_right, vp)

                # draw lane
                frame = l_left_vp.draw (frame)
                frame = l_right_vp.draw (frame)

        # combine each view
        if view_frame is None :
            view_frame = frame
        else :
            view_frame = np.hstack ((frame, view_frame))

    frame = view_frame
    # put text of iterator
    loc = (20, frame.shape[0]-20)
    cv2.putText (frame, 'Frame - {}'.format (ctr + 1), loc, cv2.FONT_HERSHEY_PLAIN, 3, (0, 128, 128), 4)

    # show image
    cv2.imshow ('default', frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break

    ctr += 1
