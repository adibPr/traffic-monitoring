#!/usr/bin/env python2

# sys module
from __future__ import division, print_function
import os
import sys

# third parties module
import cv2

# local module
path_this = os.path.abspath (os.path.dirname (__file__))

# based on MOG2
class BackgroundModel (object) :


    def __init__ (self, iterator, **kwargs) :
        self._iterator = iterator
        self._kwargs = kwargs

    def learn (self, tot_frame_init=300) : 
        self.bg_model = cv2.createBackgroundSubtractorMOG2(
                detectShadows=self._kwargs.get ('detectShadows', True)
            )

        for i in range (tot_frame_init) : 
            print ('{:2d} %'.format (int (i/tot_frame_init * 100)), end='\r')
            img = next (self._iterator)
            self.bg_model.apply (img)
            sys.stdout.flush ()

    def apply (self, img) : 
        return self.bg_model.apply (img)

    def get_background (self) : 
        return self.bg_model.getBackgroundImage ()

if __name__ == '__main__' : 
    from iterator import FrameIterator

    fi = FrameIterator ('./data/sample/session0_center.avi')
    bg = BackgroundModel (fi)
    bg.learn ()

    img = next (fi)
    fg = bg.apply (img)

    cv2.imshow ('default', fg)
    while True :
        if (cv2.waitKey(1) & 0xFF == ord('q')) :
            break

    cv2.destroyAllWindows ()
