#!/usr/bin/env python2

from __future__ import print_function, division
import math
import cv2
import numpy as np
from numpy.linalg import norm

# POINT ALWAYS IN (X,Y) DIRECTION
castInt  = lambda p : tuple ([int (_) for _ in p])

class Line (object) :

    def __init__ (self, m, b) :
        self.m = m
        self.b = b

    def get_intersection (self, line) :
        if self.m is None :
            # ideal line
            # then the b is the x
            x = self.b
            y = (line.m * x) + line.b

        elif line.m is None :
            x = line.b
            y = (self.m * x) + self.b

        else :
            x = (line.b - self.b) / (self.m - line.m)
            y = (self.m * x) + self.b

        return (x,y)

    def draw (self, frame, color=(0,255,0), size=3) :
        if len (frame.shape) == 3 :
            height, width, channel = frame.shape
        else :
            height, width = frame.shape

        # since point and all line work in (x,y) manner, then
        # find its intersection with x axis (x = 0)
        frame_c = frame.copy ()

        if self.m is None :
            # ideal line
            p1 = [self.b, 0]
            p2 = [self.b, height]

        else :
            p1 = [-self.b / self.m, 0]
            p2 = [(height - self.b) / self.m, height]

        cv2.line (frame_c, castInt (p1) , castInt (p2), color, size)
        return frame_c

    @staticmethod
    def from_two_points (pa, pb) :
        # first get line equation
        # get m
        if pa[0] is None :
            return Line (None, pb[0])

        elif pb[0] is None :
            return Line (None, pa[0])

        elif pb[0] - pa[0] == 0 :
            return Line (None, pb[0]) # its infinity

        # m = y2-y1 / x2-x1, the slope
        m = (pb[1] - pa[1]) / (pb[0] - pa[0])

        # get b
        b = pb[1] - (pb[0] * m)

        return Line (m, b)


def get_extreme_tan_point (point, contour, axis=0) :
    # get right and left (or up and down) tangent line from a given point
    # its extreme tangent line is tangent line that has biggest or smallest angle
    # assume point in (x,y) space
    # axis is either 0 (x), or 1 (y)

    min_angle = None
    max_angle = None
    tan_point_left = None
    tan_point_right = None
    # right_most_point = find_right_most_point (contour)

    if point is None :
        # mean its principle point
        for c in contour :
            c = c[0]
            if tan_point_right is None or c[axis] > tan_point_right[axis] :
                tan_point_right = c

            if tan_point_left is None or c[axis] < tan_point_left[axis] :
                tan_point_left = c

    else :
        # for each contour point
        for c in contour :
            c = c[0]

            # compute its angle
            angle = math.atan2 (c[1] - point[1], c[0] - point[0])

            # some transformation IDK
            # if angle < 0 and right_point[0] < point[0] :
            #     print ("HERE")
            #     angle = angle + (2 * math.pi)

            # if its max, save
            if max_angle is None or max_angle < angle :
                max_angle = angle
                tan_point_right = c

            # if its lowest, save
            if min_angle is None or min_angle > angle :
                min_angle = angle
                tan_point_left = c

    return tan_point_left, tan_point_right

def get_extreme_tan_point_contours (point, contours, axis=0) : 
    # get right and left (or up and down) tangent line from a
    # given point of a collection of countours
    # its extreme tangent line is tangent line that has biggest or smallest
    # angle
    # assume point in (x,y) space
    # axis is either 0 (x), or 1 (y)

    min_angle = None
    max_angle = None
    tan_point_left = None
    tan_point_right = None
    # right_most_point = find_right_most_point (contour)

    if point is None :
        # mean its principle point
        for c in contour :
            c = c[0]
            if tan_point_right is None or c[axis] > tan_point_right[axis] :
                tan_point_right = c

            if tan_point_left is None or c[axis] < tan_point_left[axis] :
                tan_point_left = c

    else :
        for contour in contours : 
            # for each contour point
            for c in contour :
                c = c[0]

                # compute its angle
                angle = math.atan2 (c[1] - point[1], c[0] - point[0])

                # some transformation IDK
                # if angle < 0 and right_point[0] < point[0] :
                #     print ("HERE")
                #     angle = angle + (2 * math.pi)

                # if its max, save
                if max_angle is None or max_angle < angle :
                    max_angle = angle
                    tan_point_right = c

                # if its lowest, save
                if min_angle is None or min_angle > angle :
                    min_angle = angle
                    tan_point_left = c

    return tan_point_left, tan_point_right

def find_right_most_point (contour) :
    # initial point
    right_point = contour[0][0]

    for c in contour :
        c = c[0]
        if c[0] > right_point[0] :
            right_point = c

    return right_point

def draw_bounding_box_contours_3D (frame, contour, vp1, vp2) :
    frame = frame.copy ()

    # draw VP 1
    color_vp1 = (0, 0, 255)
    tan_point_left, tan_point_right = get_extreme_tan_point (vp1, contour)

    line_vp1_left = Line.from_two_points (tan_point_left, vp1)
    line_vp1_right = Line.from_two_points (tan_point_right, vp1)
    # frame = line_vp1_left.draw (frame, color_vp1)
    # frame = line_vp1_right.draw (frame, color_vp1)

    # draw VP 2
    color_vp2 = (0, 255, 0)
    tan_point_left, tan_point_right = get_extreme_tan_point (vp2, contour)

    line_vp2_left = Line.from_two_points (tan_point_left, vp2)
    line_vp2_right = Line.from_two_points (tan_point_right, vp2)
    # frame = line_vp2_left.draw (frame, color_vp2)
    # frame = line_vp2_right.draw (frame, color_vp2)


    # draw VP 3
    color_vp3 = (255, 0, 0)
    tan_point_left, tan_point_right = get_extreme_tan_point (None, contour)


    line_vp3_left = Line(None, tan_point_left[0])
    line_vp3_right = Line (None, tan_point_right[0])
    # frame = line_vp3_left.draw (frame, color_vp3)
    # frame = line_vp3_right.draw (frame, color_vp3)

    intersection = []
    # A
    intersection.append (line_vp1_right.get_intersection (line_vp2_left))
    # B
    intersection.append (line_vp2_left.get_intersection (line_vp3_right))
    # C
    intersection.append (line_vp1_right.get_intersection (line_vp3_left))
    # D
    intersection.append (line_vp2_right.get_intersection (line_vp3_left))
    # F
    intersection.append (line_vp1_left.get_intersection (line_vp3_right))

    # E
    # get DU line
    line_DU = Line.from_two_points (intersection[3], vp1)
    line_AW = Line.from_two_points (intersection[0], (None, None))
    line_FV = Line.from_two_points (intersection[-1], vp2)
    ED = line_DU.get_intersection (line_AW)
    EF = line_FV.get_intersection (line_AW)

    # find the maximum distance
    if norm (np.array (ED)-np.array (intersection[0])) > norm (np.array (intersection[0])-np.array (EF)) :
        intersection.append (ED)
    else :
        intersection.append (EF)

    # G
    line_DV = Line.from_two_points (intersection[3], vp2)
    line_FU = Line.from_two_points (intersection[4], vp1)
    intersection.append (line_DV.get_intersection (line_FU))

    # H
    line_CV = Line.from_two_points (intersection[2], vp2)
    line_BU = Line.from_two_points (intersection[1], vp1)
    intersection.append (line_CV.get_intersection (line_BU))


    # drawing process
    label = 'ABCDFEGH'
    for i_idx, i in enumerate (intersection) :
        frame = cv2.circle (frame, castInt (i), 2, (255,255,255), -1)
        # cv2.putText (frame, label[i_idx], castInt(i), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

    # draw line
    pairs = 'AB, AE, AC, BF, BH, FE, FG, ED, DC, DG, CH, HG'.split (',')
    pairs = [_.strip () for _ in pairs]
    for p in pairs :
        pa = intersection[label.index (p[0])]
        pb = intersection[label.index (p[1])]
        cv2.line (frame, castInt (pa) , castInt (pb), (0, 13, 231), 2)


    return frame

def draw_bounding_ground (frame, contour, vp1, vp2) :
    frame = frame.copy ()

    # draw VP 1
    color_vp1 = (0, 0, 255)
    tan_point_left, tan_point_right = get_extreme_tan_point (vp1, contour)

    line_vp1_left = Line.from_two_points (tan_point_left, vp1)
    line_vp1_right = Line.from_two_points (tan_point_right, vp1)
    # frame = line_vp1_left.draw (frame, color_vp1)
    # frame = line_vp1_right.draw (frame, color_vp1)

    # draw VP 2
    color_vp2 = (0, 255, 0)
    tan_point_left, tan_point_right = get_extreme_tan_point (vp2, contour)

    line_vp2_left = Line.from_two_points (tan_point_left, vp2)
    line_vp2_right = Line.from_two_points (tan_point_right, vp2)
    # frame = line_vp2_left.draw (frame, color_vp2)
    # frame = line_vp2_right.draw (frame, color_vp2)


    # draw VP 3
    color_vp3 = (255, 0, 0)
    tan_point_left, tan_point_right = get_extreme_tan_point (None, contour)


    line_vp3_left = Line(None, tan_point_left[0])
    line_vp3_right = Line (None, tan_point_right[0])
    # frame = line_vp3_left.draw (frame, color_vp3)
    # frame = line_vp3_right.draw (frame, color_vp3)

    intersection = []
    # A
    intersection.append (line_vp1_right.get_intersection (line_vp2_left))
    # B
    intersection.append (line_vp2_left.get_intersection (line_vp3_right))
    # C
    intersection.append (line_vp1_right.get_intersection (line_vp3_left))
    # D
    intersection.append (line_vp2_right.get_intersection (line_vp3_left))
    # F
    intersection.append (line_vp1_left.get_intersection (line_vp3_right))

    # E
    # get DU line
    line_DU = Line.from_two_points (intersection[3], vp1)
    line_AW = Line.from_two_points (intersection[0], (None, None))
    line_FV = Line.from_two_points (intersection[-1], vp2)
    ED = line_DU.get_intersection (line_AW)
    EF = line_FV.get_intersection (line_AW)

    # find the maximum distance
    if norm (np.array (ED)-np.array (intersection[0])) > norm (np.array (intersection[0])-np.array (EF)) :
        intersection.append (ED)
    else :
        intersection.append (EF)

    # G
    line_DV = Line.from_two_points (intersection[3], vp2)
    line_FU = Line.from_two_points (intersection[4], vp1)
    intersection.append (line_DV.get_intersection (line_FU))

    # H
    line_CV = Line.from_two_points (intersection[2], vp2)
    line_BU = Line.from_two_points (intersection[1], vp1)
    intersection.append (line_CV.get_intersection (line_BU))


    # drawing process
    label = 'ABCDFEGH'
    """
    for i_idx, i in enumerate (intersection) :
        frame = cv2.circle (frame, castInt (i), 2, (255,255,255), -1)
        # cv2.putText (frame, label[i_idx], castInt(i), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    """

    # draw line
    # pairs = 'AB, AE, AC, BF, BH, FE, FG, ED, DC, DG, CH, HG'.split (',')
    pairs = 'AB, AC, CH, BH'.split (',')
    pairs = [_.strip () for _ in pairs]
    for p in pairs :
        pa = intersection[label.index (p[0])]
        pb = intersection[label.index (p[1])]
        cv2.line (frame, castInt (pa) , castInt (pb), (0, 13, 231), 2)


    return frame
