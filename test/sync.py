#!/usr/bin/env python

import os
import sys

import cv2

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..'))


from iterator import FrameIterator
from background import BackgroundModel
from util import *

session = {
    0 : {
        'center' : 569,
        'right' : 95,
        'left' : 420
    }
}

ctr = 0
while True : 
    frame = None

    # load image from each view
    for view in session[0] : 
        fpath = '../data/sample/session0_{}/{:04d}.jpg'.format (
                view, 
                session[0][view] + ctr
            )

        view_frame = cv2.imread (fpath, 1)

        # combine each view
        if frame is None : 
            frame = view_frame
        else : 
            frame = np.hstack ((frame, view_frame))

    # show image
    cv2.imshow ('default', frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break

    ctr += 1
