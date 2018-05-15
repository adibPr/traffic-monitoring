#!/usr/bin/env python2

# sys module
from __future__ import division, print_function
import os
import sys

# third parties module
import cv2

# local module
path_this = os.path.abspath (os.path.dirname (__file__))
from util import process_morphological

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
            img = next (self._iterator)
            self.bg_model.apply (img)
            sys.stdout.flush ()

    def apply (self, img) : 
        return self.bg_model.apply (img)

    def get_background (self) : 
        return self.bg_model.getBackgroundImage ()

class FrameDifference  (object) : 


    def __init__ (self, f0=None, f1=None) : 
        self.prevs = [f0, f1]

    def apply (self, image, iterations=1) : 

        prev_intersect = cv2.threshold (cv2.absdiff (self.prevs[1], self.prevs[0]), 25, 255, cv2.THRESH_BINARY)[1]
        next_intersect = cv2.threshold (cv2.absdiff (image, self.prevs[1]), 25, 255, cv2.THRESH_BINARY)[1]
        P1 = cv2.bitwise_and (prev_intersect, next_intersect)
        prev_intersect_dilate = process_morphological (prev_intersect, iterations) 
        next_intersect_dilate = process_morphological (next_intersect, iterations)
        P2 = cv2.bitwise_and (prev_intersect_dilate, next_intersect_dilate)

        # update 3FD for tsi
        self.prevs[0] = self.prevs[1]
        self.prevs[1] = image

        return P2

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
