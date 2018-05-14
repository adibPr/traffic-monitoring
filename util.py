#!/usr/bin/env python2

from __future__ import print_function, division
import os
import sys
import json

import cv2
import numpy as np

# local module
path_this = os.path.abspath (os.path.dirname (__file__))
from iterator import FrameIterator


class VPLoader (object) :

    def __init__ (self, path=os.path.join (path_this, 'data', 'gt' )) : 
        self.VP = [] 
        for ses_id in range (7) : 
            self.VP.append ({})
            for view in (("left", "right", "center")) : 
                path_vp = os.path.join (
                        path, 
                        '2016-ITS-BrnoCompSpeed/results/session{}_{}/system_dubska_bmvc14.json'.format (ses_id, view)
                    )

                # load vp
                with open (path_vp, 'r') as f_buff :
                    """
                    vp has 2 keys
                    - cars, list of cars detected, its frame and posX and posY
                    - camera_calibration, the calibration parameter result (pp, vp1, and vp2)
                    """
                    vp = json.load (f_buff)
                    self.VP[-1][view] = {
                            'vp1' : vp['camera_calibration']['vp1'],
                            'vp2' : vp['camera_calibration']['vp2']
                        }

    def get_session (self, ses_id) : 
        return self.VP[ses_id]


class MaskLoader (object) : 

    def __init__ (self, path=os.path.join (path_this, 'data', 'gt' )) : 
        self.mask = [] 
        for ses_id in range (7) : 
            self.mask.append ({})
            for view in (("left", "right", "center")) : 
                path_mask = os.path.join (
                        path, 
                        '2016-ITS-BrnoCompSpeed/dataset/session{}_{}/video_mask.png'.format (ses_id, view)
                    )

                self.mask[-1][view]  = cv2.imread (path_mask, 0)

    def get_session (self, ses_id, as_color=False) : 
        if as_color is True : 
            cur_mask = {}
            for view in (("left", "right", "center")) : 
                cur_mask[view] = cv2.cvtColor (self.mask[ses_id][view], cv2.COLOR_GRAY2BGR)  

            return cur_mask
        else : 
            return self.mask[ses_id]


class FrameIteratorLoader (object) : 

    @staticmethod
    def get_session (ses_id) : 
        fi = {}
        for view in (("left", "right", "center")) : 
            fi[view] = FrameIterator ('../../data/sync_25fps/session{}_{}'.format (ses_id, view))

        return fi



def process_morphological (fg_binary) :
    # morphological kernel
    kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    kernel10 = cv2.getStructuringElement (cv2.MORPH_RECT, (10, 10))

    # noise removal
    result = cv2.morphologyEx(fg_binary, cv2.MORPH_OPEN, kernel3)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel3)

    result = cv2.dilate (result, kernel10 , iterations=2)

    return result

def get_contours (frame_binary, min_area=200, min_width=50) :
    contours_selected = []

    im2, contours, hierarchy = cv2.findContours(
            frame_binary, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )

    # select contour that greater than the specified $min_area and
    # not part of other contour (specified by its hierarchy)
    # additionally, its width must be greater than $min_width
    for c_idx, c in enumerate (contours) :
        (x,y, w, h) = cv2.boundingRect (c)
        if cv2.contourArea(c) > min_area and hierarchy[0][c_idx][3] < 0 and\
                w > min_width and h > min_width:
            contours_selected.append (c)

    return contours_selected

def draw_bounding_box_contours (frame, contours) :
    frame_cp = frame.copy ()
    for c_idx, c in enumerate (contours) :
        (x,y, w, h) = cv2.boundingRect (c)
        cv2.rectangle (frame_cp, (x,y), (x+w, y+h), (0, 255, 0), 2)
    return frame_cp

if __name__ == '__main__' : 
    def test_VPLoader () : 
        VL = VPLoader ()
        print (VL.get_session (2))

    def test_MaskLoader () : 
        cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)
        ML = MaskLoader ()
        cv2.imshow ('default', ML.get_session (5)['right'])

        cv2.waitKey (0)

    # test_VPLoader ()
    test_MaskLoader ()
