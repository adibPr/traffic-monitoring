#!/usr/bin/env python

import os
import sys

import cv2

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..'))


from iterator import FrameIterator
from util import *

session_id = 'session0_left'

mask_path = '../data/gt/2016-ITS-BrnoCompSpeed/dataset/{}/video_mask.png'.format (session_id)
mask = cv2.imread (mask_path, 0)

fi = FrameIterator ('../data/sample/{}'.format (session_id))

prev_img = [next (fi), next (fi)]

ctr_frame = 2
while True :
    img = next (fi)

    # frame diff prev-2, prev-1, and curr frame
    prev_diff = cv2.absdiff (prev_img[-2], prev_img[-1])
    curr_diff = cv2.absdiff (prev_img[-2], img)
    frame = prev_diff + curr_diff

    frame = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.threshold (frame, 50.0, 255.0, cv2.THRESH_BINARY)[1]

    frame_clean = process_morphological (frame)
    blobs = get_contours (frame_clean)
    frame_view = draw_bounding_box_contours (prev_img[1], blobs)

    frame = cv2.cvtColor (frame_clean, cv2.COLOR_GRAY2BGR)
    frame = np.hstack ((frame, frame_view))


    # put text of iterator
    loc = (20, frame.shape[0]-20)
    cv2.putText (frame, 'Frame - {}'.format (ctr_frame), loc, cv2.FONT_HERSHEY_PLAIN, 3, (0, 128, 128), 4)

    cv2.imshow ('default', frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break

    prev_img[0] = prev_img[1]
    prev_img[1] = img

    ctr_frame += 1

cv2.destroyAllWindows ()

