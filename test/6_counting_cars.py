#!/usr/bin/env python

import os
import sys
import pickle
import json
import sys

import cv2

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..'))


from iterator import FrameIterator
from background import BackgroundModel
from util import *
from geometry import get_extreme_tan_point, Line, get_extreme_tan_point_contours

session_id = 'session0_center'

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


fi = FrameIterator ('../data/sync/{}'.format (session_id))
ctr_frame = 1
while True : 
    img = next (fi)

    div_line = Line.from_two_points (vp2, (10, img.shape[0] / 2))

    frame = div_line.draw (img)

    loc = (20, img.shape[0]-20)
    cv2.putText (frame, 'Frame - {}'.format (ctr_frame), loc, cv2.FONT_HERSHEY_PLAIN, 3, (0, 128, 128), 4)

    cv2.imshow ('default', frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break
    ctr_frame += 1
