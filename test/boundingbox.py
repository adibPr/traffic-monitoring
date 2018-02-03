#!/usr/bin/env python

import os
import sys

import cv2

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..'))


from iterator import FrameIterator
from background import BackgroundModel
from util import *

session_id = 'session0_right'

mask_path = '../data/gt/2016-ITS-BrnoCompSpeed/dataset/{}/video_mask.png'.format (session_id)
mask = cv2.imread (mask_path, 0)
        
fi = FrameIterator ('../data/sample/{}'.format (session_id))
bg = BackgroundModel (fi)

print ("Learning")
bg.learn (tot_frame_init=1)
print ("Done")

ctr_frame = 2
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

    # draw bounding box
    frame = draw_bounding_box_contours (img, blobs)

    # put text of iterator
    loc = (20, img.shape[0]-20)
    cv2.putText (frame, 'Frame - {}'.format (ctr_frame), loc, cv2.FONT_HERSHEY_PLAIN, 3, (0, 128, 128), 4)

    cv2.imshow ('default', frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break

    ctr_frame += 1

cv2.destroyAllWindows ()
