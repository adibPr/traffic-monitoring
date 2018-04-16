#!/usr/bin/env python

import os
import sys
import json
import math
import json

import cv2

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..'))


from iterator import FrameIterator
from background import BackgroundModel
from util import *
from geometry import get_extreme_tan_point, Line, get_extreme_side_point, find_right_most_point

class TopBottomLine (object) : 

    def __init__ (self, shape, **kwargs) :
        """
        Defined some reserved variable
        @input:
            - shape: [Int, Int, Int], the shape of our frame. This become input
                because our calculation of  $TOT_BUCKET_THETA based on this.
        @param:
            - TOT_BUCKET_D=100, Int, total bucket of distance of the line from origin 
                (or r in polar representation)
            - TOT_BUCKET_THETA=10, Int, total bucket of degree
            - thres=3, Int, minimum frequencies of a bin to be selected as top line
            - channel=3, Int, total different view
        @output:
            -
        """
        self.kwargs = kwargs

        # some default parameter
        self.kwargs.setdefault ("TOT_BUCKET_D", 100)
        self.kwargs.setdefault ("TOT_BUCKET_THETA", 10)
        self.kwargs.setdefault (
                "RATIO_BUCKET_D",
                math.sqrt (shape[1] ** 2 + shape[0] ** 2) / self.kwargs['TOT_BUCKET_D']
            )
        self.kwargs.setdefault (
                "RATIO_BUCKET_THETA", 
                float (4) / (self.kwargs['TOT_BUCKET_THETA'] * 2)
            )
        self.kwargs.setdefault ('thres', 3) 
        self.kwargs.setdefault ('channel', 3)

        # Read as (Parameter, frequencies, points)
        self.bucket_max = [(None, -1, None), (None, -1, None)]

        # {paramter : count}
        self.bucket_freq = {}
        self.shape = shape
        self.is_found0 = False
        self.is_found1 = False
        self.prev_r = [None] * self.kwargs['channel'] # radius in real measurement (pixel)
        self.prev_params = [None] * self.kwargs['channel'] # in parameter form
        self.prev_points = [None] * self.kwargs['channel'] 

    def update0 (self, lines, points) : 
        """
        find the first line, line that correspondent to the first visible on the scene
        @input:
            - lines, [Line], minimum line from a frame that has 
            - points, [(float, float)], 3 points reference, for drawing purposes
                since lines reconstruction has messy thing
        @param: 
            -
        @output:
            -
        """

        assert len (lines) == self.kwargs['channel'], "Input lines length must be {}".format (self.channel)

        line_params = []
        polars = []
        for line in lines : 
            if line is None or line.b > self.shape[0] : 
                polar = None
                line_params.extend ((None, None))
            else : 
                polar = line.to_polar ()
                line_params.extend ((
                    int (polar[0] / self.kwargs['RATIO_BUCKET_D']), 
                    int ((polar[1] + 2) / self.kwargs['RATIO_BUCKET_THETA'])# +2 to make all positive
                ))

            polars.append (polar)

        line_params = tuple (line_params)

        # only update that doesn't have None
        if (all ([_ is not None for _ in line_params])) : 
            self.bucket_freq[line_params] = self.bucket_freq.get (line_params, 0) + 1

            if self.bucket_freq[line_params] > self.bucket_max[0][1] :
                self.bucket_max[0] = (line_params, self.bucket_freq[line_params], points)

        if self.bucket_max[0][1] >= self.kwargs['thres'] : 
            self.is_found0 = True
            self.prev_r = [_[0] for _ in polars]
            self.prev_params = line_params
            self.prev_points = points

    def update1 (self, lines, points) :
        """
        find the second line, line that correspondent to the maximum line before
        the synced line vanished
        @input:
            - lines, [Line], minimum line from a frame that has 
            - points, [(float, float)], 3 points reference, for drawing purposes
                since lines reconstruction has messy thing
        @param: 
            -
        @output:
            -
        """
        # maximum is prev lines
        this_r = []
        line_params = []
        # if exist lines that current lines is None, or 
        for l_idx, line in enumerate (lines) : 
            #FIXME
            if line is None or points[l_idx][0] < 1 or points[l_idx][1] >= self.shape[0] - 10 :
                self.bucket_max[1] = (self.prev_params, 0, self.prev_points)
                self.is_found1 = True
                return

            else : 
                polar = line.to_polar ()

                line_params.extend ((
                    int (polar[0] / self.kwargs['RATIO_BUCKET_D']), 
                    int ((polar[1] + 2) / self.kwargs['RATIO_BUCKET_THETA'])# +2 to make all positive
                ))

                this_r.append (polar[0])

        # this r is lower than previous r, then
        # in here lower means that if the difference is in one sign
        diff_r = [this_r[i] - self.prev_r[i] for i in range (len (this_r))]

        # so if not, then we found it
        if not (all ([_ < 0 for _ in diff_r]) or all ([_ >= 0 for _ in diff_r])) :
            self.bucket_max[1] = (self.prev_params, 0, self.prev_points)
            self.is_found1 = True
            return
        else : 
            # update for next iteration
            self.prev_r = this_r
            self.prev_params = line_params
            self.prev_points = points

    def update (self, lines, points) : 
        """
        Adjust parameter based on new update of all channel line
        @input:
            - lines, [Line], minimum line from a frame that has 
            - points, [(float, float)], 3 points reference, for drawing purposes
                since lines reconstruction has messy thing
        @param: 
            -
        @output:
            -
        """
        if not self.is_found0 : 
            self.update0 (lines, points)
        else : 
            if not self.is_found1 : 
                self.update1 (lines, points)

    def inverse_bin (self, b) : 
        """
        Convert back from bin index into r-theta approx values
        @input:
            - b, [Int, Int], bin index
        @param:
            -
        @output:
            - (r, theta), (float, float) the line parameter
        """

        r, theta = b[0] * self.kwargs['RATIO_BUCKET_D'], b[1] * self.kwargs['RATIO_BUCKET_THETA'] - 2
        return (r, theta)

    def inverse_bin_Line (self, b, point):
        """
        Convert back from bin index into a Line, drawn from a point
        @input:
            - b, [Int, Int], bin index
            - point, [Float, Float], (x,y) value coordinate, a reference point to
                create a Line
        @param:
            -
        @output:
            - line, a Line object
        """

        r, theta = self.inverse_bin (b)
        x = r * math.cos (theta)
        y = r * math.sin (theta)
        l = Line.from_two_points ((x,y), point)

        return l


class RightLeftLine (object) : 

    def __init__ (self, vp1) :
        """
        Defined some reserved variable
        @input: 
            - vp1, (float, float), the position of the first vanishing point of this view
        @param:
            -
        @output:
            -
        """

        self.vp1 = vp1

        # for storing extreme point and angle
        # format [angle, point]
        self.max = [None, None]
        self.min = [None, None]

    def update (self, cps) : 
        """
        Update internal parameter based on incoming blobs
        @input: 
            - cps, [[point]], corner point of blobs
        @param:
            -
        @output:
            -
        """

        curr_max = [None, None]
        curr_min = [None, None]

        for cp in cps:
            # max-min for right-left
            for c_idx, c in enumerate (cp) : 
                # compute angle
                ang = math.atan2 (c[1] - self.vp1[1], c[0] - self.vp1[0])

                # compare with  global  value
                if self.max[0] is None or self.max[0] < ang : 
                    self.max[0] = ang
                    self.max[1] = c

                if self.min[0] is None or self.min[0] > ang : 
                    self.min[0] = ang
                    self.min[1] = c

    def draw (self, frame, **kwargs) : 
        """
        draw left and right max line
        @input:
            - frame
        @param: 
            - color
        @output:
            - frame
        """


        frame = frame.copy ()

        lines = []
        # first line min
        if self.min[1] is not None : 
            lines.append (Line.from_two_points (self.min[1], self.vp1))

        # then line max
        if self.max[1] is not None : 
            lines.append (Line.from_two_points (self.max[1], self.vp1))

        for l in lines : 
            frame = l.draw (frame, **kwargs)
        return frame


def get_corner_points (blob, vp1, vp2, pos_vp1, pos_vp2) : 
    cps = []
    for b in blobs : 
        # first get from left to right
        c1_right, c1_left = get_extreme_tan_point (vp1, b)
        c2_right, c2_left = get_extreme_tan_point (vp2, b)
        c3_right, c3_left = get_extreme_tan_point (None, b)

        # get ground line vp1
        # if position of vp1 is top-left (0), then 
        # the ground line is vp1 and c1_left
        if vp1_pos == 0 :
            gl_vp1 = Line.from_two_points (vp1, c1_left)

        # otherwise the groundline is vp1 and c1_right
        else :
            gl_vp1 = Line.from_two_points (vp1, c1_right) 

        # get ground line vp2
        # same with above, if the position of vp2  is top-right (1)
        # then the ground line is line between vp2 and c2_right
        if vp2_pos == 1 : 
            gl_vp2 = Line.from_two_points (vp2, c2_right)

        # otherwise it between vp2 and c2_left
        else :
            gl_vp2 = Line.from_two_points (vp2, c2_left)

        # get ground line vp3
        # it depend on the vp1, if vp1 position is top left, 
        # then construct line with c3_left
        if vp1_pos == 0 : 
            gl_vp3 = Line (None, c3_left[0])

        # otherwise with c3_right
        else : 
            gl_vp3 = Line (None, c3_right[0])

        """
        # get helper line
        line = [
                Line.from_two_points (vp1, c1_left),
                Line.from_two_points (vp2, c2_right),
                Line (None, c3_left[0])
            ]
        """

        # get corner point 
        cp = [
                gl_vp1.get_intersection (gl_vp2),
                gl_vp2.get_intersection (gl_vp3)
            ]

        cps.append (cp)
    return cps

def get_position (vp, shape) : 
    """
    Get region position of a vp
    Position
    _________
    | 0 | 1 |
    ---------
    | 2 | 3 |
    ---------
    """

    if vp[0] > (shape[1] / 2) : 
        if vp[1] > (shape[0]/2) : 
            pos = 3 
        else : 
            pos = 1
    else : 
        if vp[1] > (shape[0]/2) : 
            pos = 2
        else : 
            pos = 0

    return pos

VP = VPLoader ()
MASK = MaskLoader ()
VIEW = ("right", "center", "left")

cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)

for ses_id in range (5, 6) : 

    # some variable
    vp = VP.get_session (ses_id)
    fi = FrameIteratorLoader.get_session (ses_id)
    masks = MASK.get_session (ses_id)

    # initialing prev blobs and RightLeftLine object
    prev_img = {}
    prev_img_color= {}
    RLL = {} 

    for view in VIEW : 

        RLL[view] = RightLeftLine (vp[view]['vp1'])
        
        prev_img[view] = [None, None]
        prev_img_color[view] = [None, None]

        for i in range (2) : 
            img = next (fi[view])
            img_color = img.copy ()
            img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

            # save background
            prev_img[view][i] = img 
            prev_img_color[view][i] = img_color


    # mask for all corner point detected
    corner_mask = [np.zeros (masks[view].shape), np.zeros (masks[view].shape), np.zeros (masks[view].shape)]

    # mask for all top bottom line detected
    shape_3d = list (masks[view].shape)
    shape_3d.append (3)
    shape_3d = tuple (shape_3d)
    topbot_mask = [np.zeros (shape_3d), np.zeros (shape_3d), np.zeros (shape_3d)]

    kernel10 = cv2.getStructuringElement(cv2.MORPH_CROSS,(10,10))

    TBL = TopBottomLine (shape_3d)

    ctr = 0
    while True :
        view_frame = None
        prev_frame = {}
        for view in VIEW : 
            prev_frame[view] =  None

        # for holding line and point paramter
        # for top bottom
        line_extreme_vp2 = []
        point_extreme_vp2 = []

        # load image from each view
        for view_idx, view in enumerate (VIEW) :
            img = next (fi[view])
            img_color = img.copy ()
            img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

            # vanishing point and its position
            vp1 = vp[view]['vp1']
            vp2 = vp[view]['vp2']
            vp1_pos = get_position (vp1, img.shape)
            vp2_pos = get_position (vp2, img.shape)

            # by 3 frame difference
            prev_intersect = cv2.threshold (cv2.absdiff (prev_img[view][1], prev_img[view][0]), 25, 255, cv2.THRESH_BINARY)[1]
            next_intersect = cv2.threshold (cv2.absdiff (img, prev_img[view][1]), 25, 255, cv2.THRESH_BINARY)[1]

            # by 3 frame of "Vehicle speed measurement based on gray constraint optical flow algorithm"
            P1 = cv2.bitwise_and (prev_intersect, next_intersect)
            prev_intersect_dilate = process_morphological (prev_intersect) 
            next_intersect_dilate = process_morphological (next_intersect)
            
            P2 = cv2.bitwise_and (prev_intersect_dilate, next_intersect_dilate)
            fg_3frame = cv2.bitwise_xor (P1, P2)
            fg = P2

            # remove noise
            fg = process_morphological (fg)

            # apply mask
            fg = cv2.bitwise_and (fg, masks[view])

            # get blobs
            blobs = get_contours (fg)

            frame = prev_img_color[view][1] # why one ? because [prev, cur], next
            # frame = cv2.cvtColor (fg, cv2.COLOR_GRAY2BGR)
            # frame = draw_bounding_box_contours (frame, blobs)

            corner_points = get_corner_points (blobs, vp1, vp2, vp1_pos, vp2_pos)

            # update left right corner points
            RLL[view].update (corner_points)

            # draw-right-left
            # frame = RLL[view].draw (frame)

            # for vp2 this frame only
            min_angle_vp2 = None
            min_point_vp2 = None

            for cp in corner_points : 
                # for c in cp :
                #     cv2.circle (frame, tuple ([int (_) for _ in c]), 50, (255, 255, 10), thickness=-1)

                # cp is pair of corner point, we just need one
                # we use again corner point detected, but since
                # all cp will be have same angle toward vp2, then we just use 
                # corner point that has the highest (y) axis, or with the lower (x)
                # depend which one suites since it will be used
                # to determine stop function
                curr_y = 0
                for cc in cp : 
                    if cc[0] < 1 : # near y axis, touching the border
                        c = cc
                        break
                    elif cc[1] > curr_y :
                        c = cc
                        curr_y = c[1]

                # compute angle
                # this depend on the vp2 position, if position is 1, use it
                # otherwise negative it
                ang = math.atan2 (c[1] - vp2[1], c[0] - vp2[0])
                if vp2_pos == 1 : 
                    ang = -ang

                # if the first line is found, then for the next, using the
                # furthest line
                if TBL.is_found0 : 
                    if min_angle_vp2 is None or min_angle_vp2 < ang : 
                        min_angle_vp2 = ang
                        min_point_vp2 = c
                else : 
                    if min_angle_vp2 is None or min_angle_vp2 > ang : 
                        min_angle_vp2 = ang
                        min_point_vp2 = c

            # convert max top-bottom from point to polar line
            if min_point_vp2 is not None : 
                # first find its line parameter
                # just take the furthest (bottom)
                l = Line.from_two_points (vp2, min_point_vp2)
                line_extreme_vp2.append (l)

                # draw top-bottom 
                frame = l.draw (frame)
                point_extreme_vp2.append (min_point_vp2)

            else : 
                line_extreme_vp2.append (None)
                point_extreme_vp2.append (None)


            # combine each view
            if view_frame is None :
                view_frame = frame
            else :
                view_frame = np.hstack ((frame, view_frame))

            prev_frame[view] = frame.copy ()

            # update frame difference
            prev_img[view][0] = prev_img[view][1]
            prev_img[view][1] = img

            prev_img_color[view][0] = prev_img_color[view][1]
            prev_img_color[view][1] = img_color

        frame = view_frame

        if not (TBL.is_found0 and TBL.is_found1) : 
            TBL.update (line_extreme_vp2, point_extreme_vp2)

        else : 
            # assumption the right and left is already stable
            if ctr <= 500 : 
                # saving the result
                # result = {"session{}".format (ses_id) : {}}

                # draw top  bottom
                for t in TBL.bucket_max : 
                    for i, v in enumerate (VIEW) : 
                        top_point = t[2][i]
                        l = Line.from_two_points (top_point, vp[v]['vp2'])
                        prev_frame[v] = l.draw (prev_frame[v], color=(0,0,255))

                        # for left and right
                        max_line = Line.from_two_points (vp[v]['vp1'], RLL[v].max[1])
                        min_line = Line.from_two_points (vp[v]['vp1'], RLL[v].min[1])

                        # draw left right
                        # prev_frame[v] = max_line.draw (prev_frame[v], color=(0,0,255))
                        # prev_frame[v] = min_line.draw (prev_frame[v], color=(0,0,255))


                        """
                        # saving result
                        result['session{}'.format (ses_id)][view] = [
                                    # top-left, intersection of  max and the first
                                    max_line.get_intersection (line_result[view_idx]), 
                                    # top-right,  intersection of min and the first
                                    min_line.get_intersection (line_result[view_idx]),
                                    # bot-left, intersection of max and the second
                                    max_line.get_intersection (line_result[3 + view_idx]),
                                    # bot-right, intersection of min adn the second
                                    min_line.get_intersection (line_result[3 + view_idx])
                                ]

                with open ('result/500-iteration.json', 'w') as f_json : 
                    json.dump (result, f_json)
                """

                for i, view in enumerate (VIEW) : 
                    mask_color = cv2.cvtColor (masks[view], cv2.COLOR_GRAY2BGR)
                    prev_frame[view] = cv2.addWeighted (prev_frame[view], 0.7, mask_color, 0.2, 0)
                    cv2.imwrite ('result/common_ground/{}-{}.jpg'.format (ses_id, view), prev_frame[view])

                break


        # drawing
        # put text of iterator
        loc = (20, frame.shape[0]-20)
        cv2.putText (frame, 'Frame - {}'.format (ctr + 1), loc, cv2.FONT_HERSHEY_PLAIN, 3, (0, 128, 128), 4)

        # show image
        cv2.imshow ('default', frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')) :
            break

        ctr += 1
        print (ctr)
