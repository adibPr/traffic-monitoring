#!/usr/bin/env python

import os
import sys
import pickle
import json

import cv2

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..'))


from iterator import FrameIterator
from background import BackgroundModel
from util import *
from geometry import get_extreme_tan_point, Line, get_extreme_tan_point_contours

session_id = 'session0_right'

mask_path = '../data/gt/2016-ITS-BrnoCompSpeed/dataset/{}/video_mask.png'.format (session_id)
mask = cv2.imread (mask_path, 0)


vp_path = '../data/gt/2016-ITS-BrnoCompSpeed/results/{}/system_dubska_bmvc14.json'.format (session_id)
with open (vp_path, 'r') as f_buff :
    """
    vp has 2 keys
    - cars, list of cars detected, its frame and posX and posY
    - camera_calibration, the calibration parameter result (pp, vp1, and vp2)
    """
    vp = json.load (f_buff)
vp1 = vp['camera_calibration']['vp1']
vp2 = vp['camera_calibration']['vp2']


fi = FrameIterator ('../data/sample/{}'.format (session_id))
bg = BackgroundModel (fi)

print ("Learning")
bg.learn (tot_frame_init=1)
print ("Done")

ctr_frame = 0

# what I want is the function to find most left tangen and right tangen of
# given blob list and a point
while True :
    img = next (fi)
    frame = img
    fg = bg.apply (img)

    # remove shadows, i.e value 127
    fg = cv2.threshold (fg, 200, 255, cv2.THRESH_BINARY)[1]
    # remove noise
    fg = process_morphological (fg)
    # apply mask
    fg = cv2.bitwise_and (fg, mask)
    # get blobs
    blobs = get_contours (fg)

    # only consider that has object on it
    if blobs :

        for vp in (vp1, vp2) :
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

    # put text of iterator
    loc = (20, img.shape[0]-20)
    cv2.putText (frame, 'Frame - {}'.format (ctr_frame), loc, cv2.FONT_HERSHEY_PLAIN, 3, (0, 128, 128), 4)

    cv2.imshow ('default', frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break

    ctr_frame += 1
