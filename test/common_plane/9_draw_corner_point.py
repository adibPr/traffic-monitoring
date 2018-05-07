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
from geometry import get_extreme_tan_point, Line, get_extreme_side_point, find_right_most_point

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

session = {
    0 : {
        'center' : 569,
        'right' : 95,
        'left' : 420
    }
}

_id = 0

# load masks
masks = {_id : {}}
for view in session[_id] : 
    mask_path = '../data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/video_mask.png'.format (_id, view)
    masks[_id][view] = cv2.imread (mask_path, 0)

# load vanishing points
vps = {_id : {}}
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

# generate frame iterator
fi = {_id : {}}
for view in session[_id] : 
    fi[_id][view] = FrameIterator ('../data/sync/session{}_{}'.format (_id, view))

# define background model
bms = {_id : {}} 
for view in session[_id] : 
    bms[_id][view] = BackgroundModel (fi[_id][view])
    print ("Learning for session {}-{}".format (_id, view))
    bms[_id][view].learn (tot_frame_init=2)
    print ("Done")

# initialing prev blobs
prev_fg = {_id : {}}
for view in session[_id] : 
    prev_fg[_id][view] = [None, None]
    for i in range (2) : 
        img = next (fi[_id][view])

        # by background subtraction
        fg = bms[_id][view].apply (img)
        # remove shadows, i.e value 127
        fg = cv2.threshold (fg, 200, 255, cv2.THRESH_BINARY)[1]

        # remove noise
        fg = process_morphological (fg)
        # apply mask
        fg = cv2.bitwise_and (fg, masks[_id][view])
        
        # save background
        prev_fg[_id][view][i] = fg 

ctr = 0
while True :
    view_frame = None
    prev_frame = [None, None, None]

    # load image from each view
    for view_idx, view in enumerate (session[_id]) :
        img = next (fi[_id][view])

        vp1 = vps[_id][view]['vp1']
        vp2 = vps[_id][view]['vp2']

        # by background subtraction
        fg = bms[_id][view].apply (img)
        # remove shadows, i.e value 127
        fg = cv2.threshold (fg, 200, 255, cv2.THRESH_BINARY)[1]

        # remove noise
        fg = process_morphological (fg)
        # apply mask
        fg = cv2.bitwise_and (fg, masks[_id][view])

        # combine info from prev_fg
        prev_fg_intersect = cv2.bitwise_or (prev_fg[_id][view][1], prev_fg[_id][view][0])
        curr_fg_intersect = cv2.bitwise_or (fg, prev_fg[_id][view][1])
        fgs = cv2.bitwise_and (prev_fg_intersect, curr_fg_intersect)

        # get blobs
        blobs = get_contours (fg)

        # drawing
        frame = cv2.cvtColor (fgs, cv2.COLOR_GRAY2BGR)
        frame = img
        # frame = draw_bounding_box_contours (frame, blobs)
        # frame = fg

        # only consider that has object on it
        if blobs :

            for b in blobs : 

                # first get from left to right
                c1_right, c1_left = get_extreme_tan_point (vp1, b)
                c2_right, c2_left = get_extreme_tan_point (vp2, b)
                c3_right, c3_left = get_extreme_tan_point (None, b)

                # get line
                line = [
                        Line.from_two_points (vp1, c1_left),
                        Line.from_two_points (vp2, c2_right),
                        Line (None, c3_left[0])
                    ]

                for l_idx, l in enumerate (line) : 
                    color = [0, 0, 0]
                    color[l_idx] += 255
                    frame = l.draw (frame, color=color)

                # draw corner point 
                cp = [
                        line[0].get_intersection (line[1]),
                        line[1].get_intersection (line[2])
                    ]

                for c in cp : 
                    frame = cv2.circle (frame, tuple ([int (_) for _ in c]) , 10, (50,100,255), -1)

        # combine each view
        if view_frame is None :
            view_frame = frame
        else :
            view_frame = np.hstack ((frame, view_frame))

        prev_frame[view_idx] = frame.copy ()

    frame = view_frame

    # put text of iterator
    loc = (20, frame.shape[0]-20)
    cv2.putText (frame, 'Frame - {}'.format (ctr + 1), loc, cv2.FONT_HERSHEY_PLAIN, 3, (0, 128, 128), 4)

    # show image
    cv2.imshow ('default', frame)

    if ctr == 17 : 
        for f_idx, f in enumerate (prev_frame) : 
            cv2.imwrite ('out-{}.jpg'.format (f_idx), f)
        sys.exit ()

    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break
    
    ctr += 1
