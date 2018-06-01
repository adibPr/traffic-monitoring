#!/usr/bin/env python2

from __future__ import print_function, division
import os
import sys
import json

import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from geometry import Line, get_extreme_tan_point

def draw_polylines (img, corner, color=(0,0,255), thickness=5) : 
    img = img.copy ()

    cornerInt = []
    for c in corner : 
        cornerInt.append (tuple ([int (_) for _ in c]))

    corner = np.array (corner, np.int32)
    corner = corner.reshape ((-1, 1, 2))

    cv2.polylines (img, [corner], True , color, thickness)
    return img

def get_corner_ground (vp1, vp2, points) : 
    # convention of points : 
    #   [left, top, right, bottom]

    lines = [
            Line.from_two_points (vp1, points[0]), # left line
            Line.from_two_points (vp2, points[1]), # top line,
            Line.from_two_points (vp1, points[2]), # right line
            Line.from_two_points (vp2, points[3]) # bottom line
        ]

    corner = (
            lines[0].get_intersection (lines[1]), # top left corner
            lines[1].get_intersection (lines[2]), # top right corner
            lines[2].get_intersection (lines[3]), # bottom right corner
            lines[3].get_intersection (lines[0]) # bottom left corner
        )
    
    return corner

def get_contours (frame_binary, min_area=200, min_width=30) :
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
        if cv2.contourArea(c) > min_area and hierarchy[0][c_idx][3] < 0 and \
                h > min_width:
            contours_selected.append (c)

    return contours_selected

def get_middle_point (p1, p2) : 
    x1 = int ((p1[0] + p2[0]) / 2)
    x2 = int ((p1[1] + p2[1]) / 2)
    return (x1, x2)

def is_contained (blob, middle_point) : 
    (x,y, w, h) = cv2.boundingRect (blob)
    if middle_point[0] >= x and middle_point[0] <= x + w \
            and middle_point[1] >= y and middle_point[1] <=  y + h :
        return True
    return False

def get_top_polylines (bottom_polylines, blob) : 

    # for back and front 
    height = [0, 0]

    back_x  = sorted ([bottom_polylines[0][0], bottom_polylines[1][0]])
    back_y  = sorted ([bottom_polylines[0][1], bottom_polylines[1][1]])
    front_x = sorted ([bottom_polylines[2][0], bottom_polylines[3][0]])

    for point in blob : 
        x,y  = point[0]
        # for back height
        if back_x[0] <= x and x <= back_x[1] and y <= back_y[1] : 
            # check if the height is maximum
            max_height = max ([abs (y - r[1]) for r in bottom_polylines[:2]])

            if max_height > height[0] : 
                height[0] = max_height
        
        if front_x[0] <= x and x <= front_x[1] :
            # check if the height is maximum
            max_height = max ([abs (y - r[1]) for r in bottom_polylines[2:]])

            if max_height > height[1] : 
                height[1] = max_height

    # we find back height
    # to to find front height, we use ration of length x and y
    ratio =  euclidean (bottom_polylines[2], bottom_polylines[3]) / euclidean (bottom_polylines[0], bottom_polylines[1]) 
    # height[0] = max (height[0], 50)
    height[0] = 100 
    height[1] = int (ratio * height[0])
    top_polylines = [
            [bottom_polylines[0][0], bottom_polylines[0][1] - height[0]],
            [bottom_polylines[1][0], bottom_polylines[1][1] - height[0]],
            [bottom_polylines[2][0], bottom_polylines[2][1] - height[1]],
            [bottom_polylines[3][0], bottom_polylines[3][1] - height[1]]
        ]

    return top_polylines

# def get_top_polylines_VP (bottom_polylines, blob, vp1) :
#     tan_point_left, tan_point_right = get_extreme_tan_point (blob, vp1)



def draw_3D_box (frame, bottom_polylines, top_polylines, color=(255,0,0), thickness=3) : 
    frame = frame.copy ()
    frame = draw_polylines (frame, bottom_polylines, color, thickness )
    frame = draw_polylines (frame, top_polylines, color, thickness )

    for i in range (4) : 
        cv2.line (frame, tuple (top_polylines[i]), tuple (bottom_polylines[i]), color, thickness) 

    return frame

def map_point (points, M) : 
    points = np.matrix (points)
    # insert one
    ones = np.ones ((points.shape[0], 1))
    points = np.hstack ((points, ones))
    points = points.transpose ()
        
    # draw on the real image
    point_map = M.dot (points).transpose () # get inverse location of box
    point_map /= point_map[:, 2] # divide by homogoneous scale
    point_map = point_map[:, :-1].astype ('int').tolist () # convert into index
    return point_map

def get_most_left_blobs (blobs, n=1) : 
    x_pos = []
    for b_idx, b in enumerate (blobs) : 
        (x,y,w,h) = cv2.boundingRect (b)
        x_pos.append ((b_idx, x))

    x_pos = sorted (x_pos, key=lambda e: e[1])
    return [blobs[e[0]] for e in x_pos[:n]]
    

class TSI (object) : 


    def __init__ (
                self, 
                M, 
                size=(1000, 300),
                VDL_IDX=0, 
                VDL_SIZE=5, 
                VDL_SCALE=2/4,
                strip=1000
            ) : 

        self.M = M
        self.size = size
        self.VDL_IDX = VDL_IDX
        self.VDL_SIZE = VDL_SIZE
        self.VDL_SCALE = VDL_SCALE
        self.tsi = None
        self.TRACK_MAX = size[0] / VDL_SIZE

    def apply (self, image) : 
        if image.shape[0] != self.size[0] or image.shape[1] != self.size[1] : 
            dst = cv2.warpPerspective (image, self.M, self.size)

        if len (image.shape) == 3 : 
            strip = dst[:, self.VDL_IDX:self.VDL_IDX+self.VDL_SIZE, :]
        else : 
            strip = dst[:, self.VDL_IDX:self.VDL_IDX+self.VDL_SIZE]

        if self.tsi is None : 
            if len (image.shape) == 3 : 
                self.tsi = np.zeros ((self.size[1], self.size[0], 3)).astype ('uint8')
            else : 
                self.tsi = np.zeros ((self.size[1], self.size[0])).astype ('uint8')

        self.tsi = np.hstack ((strip, self.tsi))

        if len (image.shape) == 3: 
            self.tsi = self.tsi[:, :self.size[0], :]
        else : 
            self.tsi = self.tsi[:, :self.size[0]]

        return self.tsi

    def get_approximate_length (self, blob, padding=0) : 
        (x,y,w,h) = cv2.boundingRect (blob)
        x -= padding
        y -= padding
        w += padding
        h += padding

        w = int ((self.VDL_SCALE) * self.VDL_SIZE * w) #convert to actual length
        x = int (x * self.VDL_SCALE * self.VDL_SIZE) # convert x to actual length
        return (x,y,w,h)

class EPI (TSI) : 

    def apply (self, image) : 
        if image.shape[0] != self.size[0] or image.shape[1] != self.size[1] : 
            dst = cv2.warpPerspective (image, self.M, self.size)

        if len (image.shape) == 3 : 
            strip = dst[self.VDL_IDX:self.VDL_IDX+self.VDL_SIZE,: , :]
        else : 
            strip = dst[self.VDL_IDX:self.VDL_IDX+self.VDL_SIZE, :]

        if self.tsi is None : 
            if len (image.shape) == 3 : 
                self.tsi = np.zeros ((self.size[1], self.size[0], 3)).astype ('uint8')
            else : 
                self.tsi = np.zeros ((self.size[1], self.size[0])).astype ('uint8')

        self.tsi = np.vstack ((strip, self.tsi))

        if len (image.shape) == 3: 
            self.tsi = self.tsi[:self.size[1], :, :]
        else : 
            self.tsi = self.tsi[:self.size[1], :]

        return self.tsi

