#!/usr/bin/env python2

from __future__ import print_function, division

import cv2
import numpy as np

def process_morphological (fg_binary) :

    # morphological kernel
    kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    kernel10 = cv2.getStructuringElement (cv2.MORPH_CROSS, (10, 10))

    # noise removal
    result = cv2.morphologyEx(fg_binary, cv2.MORPH_OPEN, kernel3)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel3)

    result = cv2.dilate (result, kernel10 , iterations=1)

    return result

def get_contours (frame_binary, min_area=400, min_width=10) :
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
        if cv2.contourArea(c) > min_area and hierarchy[0][c_idx][3] < 0 :
            contours_selected.append (c)

    return contours_selected

def draw_bounding_box_contours (frame, contours) :
    frame_cp = frame.copy ()
    for c_idx, c in enumerate (contours) :
        (x,y, w, h) = cv2.boundingRect (c)
        cv2.rectangle (frame_cp, (x,y), (x+w, y+h), (0, 255, 0), 2)
    return frame_cp

