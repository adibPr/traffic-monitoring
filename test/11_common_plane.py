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

        self.bucket_max = [(None, -1), (None, -1)]
        self.bucket_freq = {}
        self.shape = shape
        self.is_found0 = False
        self.is_found1 = False
        self.prev_r = [None] * self.kwargs['channel'] # radius in real measurement (pixel)
        self.prev_params = [None] * self.kwargs['channel'] # in parameter form

    def update0 (self, lines) : 
        """
        find the first line, line that correspondent to the first visible on the scene
        @input:
            - lines, [Line], minimum line from a frame that has 
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
                self.bucket_max[0] = (line_params, self.bucket_freq[line_params])

        if self.bucket_max[0][1] >= self.kwargs['thres'] : 
            self.is_found0 = True
            self.prev_r = [_[0] for _ in polars]
            self.prev_params = line_params

    def update1 (self, lines) :
        """
        find the second line, line that correspondent to the maximum line before
        the synced line vanished
        @input:
            - lines, [Line], minimum line from a frame that has 
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
            if line is None or line.b > self.shape[0]: 
                self.bucket_max[1] = (self.prev_params, 0)
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
        diff_r = [this_r[i] - self.prev_r[i]]
        # so if not, then we found it
        if not (all ([_ < 0 for _ in diff_r]) or all ([_ >= 0 for _ in diff_r])) :
            self.bucket_max = (line_params, 0)
            self.is_found1 = True
            return
        else : 
            # update for next iteration
            self.prev_r = this_r
            self.prev_params = line_params

    def update (self, lines) : 
        """
        Adjust parameter based on new update of all channel line
        @input:
            - lines, [Line], minimum line from a frame that has 
        @param: 
            -
        @output:
            -
        """
        if not self.is_found0 : 
            self.update0 (lines)
        else : 
            if not self.is_found1 : 
                self.update1 (lines)

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


cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)

session = {
    0 : {
        'center' : None,
        'right' : None,
        'left' : None 
    }
}

_id = 0
# load masks
masks = {_id : {}}
for view in session[_id] : 
    mask_path = '../data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/video_mask.png'.format (_id, view)
    masks[_id][view] = cv2.imread (mask_path, 0)

# load vanishing points
vps = {_id : {}}
for view in session[_id] : 
    vp_path = '../data/gt/2016-ITS-BrnoCompSpeed/results/session{}_{}/system_dubska_bmvc14.json'.format (_id, view)

    with open (vp_path, 'r') as f_buff :
        """
        vp has 2 keys
        - cars, list of cars detected, its frame and posX and posY
        - camera_calibration, the calibration parameter result (pp, vp1, and vp2)
        """
        vp = json.load (f_buff)
        vps[_id][view] = {
                'vp1' : vp['camera_calibration']['vp1'],
                'vp2' : vp['camera_calibration']['vp2']
            }

# generate frame iterator
fi = {_id : {}}
for view in session[_id] : 
    fi[_id][view] = FrameIterator ('../data/sync_25fps/session{}_{}'.format (_id, view))

# define background model
bms = {_id : {}} 
for view in session[_id] : 
    bms[_id][view] = BackgroundModel (fi[_id][view])
    print ("Learning for session {}-{}".format (_id, view))
    bms[_id][view].learn (tot_frame_init=2)
    print ("Done")

# initialing prev blobs
prev_img = {_id : {}}
prev_img_color= {_id : {}}
for view in session[_id] : 
    prev_img[_id][view] = [None, None]
    prev_img_color[_id][view] = [None, None]
    for i in range (2) : 
        img = next (fi[_id][view])
        img_color = img.copy ()
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

        # save background
        prev_img[_id][view][i] = img 
        prev_img_color[_id][view][i] = img_color

# load json gt data
with open ('../data/sync_25fps/common_point/result.json') as f_buf : 
    common_plane_coor = json.load (f_buf)

# for storing extrema of vp1
max_angle_vp1 = [None, None, None]
max_point_vp1 = [None, None, None]
min_angle_vp1 = [None, None, None]
min_point_vp1 = [None, None, None]

# mask for all corner point detected
corner_mask = [np.zeros (masks[_id][view].shape), np.zeros (masks[_id][view].shape), np.zeros (masks[_id][view].shape)]
# mask for all top bottom line detected
shape_3d = list (masks[_id][view].shape)
shape_3d.append (3)
shape_3d = tuple (shape_3d)
topbot_mask = [np.zeros (shape_3d), np.zeros (shape_3d), np.zeros (shape_3d)]

kernel10 = cv2.getStructuringElement(cv2.MORPH_CROSS,(10,10))

TBL = TopBottomLine (shape_3d)

ctr = 0
while True :
    view_frame = None
    prev_frame = [None, None, None]

    line_extreme_vp2 = []

    # load image from each view
    for view_idx, view in enumerate (session[_id]) :
        img = next (fi[_id][view])
        img_color = img.copy ()
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

        vp1 = vps[_id][view]['vp1']
        vp2 = vps[_id][view]['vp2']

        # by MOG background subtraction
        fg_mog = bms[_id][view].apply (prev_img[_id][view][1])
        # remove shadows, i.e value 127
        fg_mog = cv2.threshold (fg_mog, 200, 255, cv2.THRESH_BINARY)[1]

        # by 3 frame difference
        prev_intersect = cv2.threshold (cv2.absdiff (prev_img[_id][view][1], prev_img[_id][view][0]), 25, 255, cv2.THRESH_BINARY)[1]
        next_intersect = cv2.threshold (cv2.absdiff (img, prev_img[_id][view][1]), 25, 255, cv2.THRESH_BINARY)[1]

        # by 3 frame of "Vehicle speed measurement based on gray constraint optical flow algorithm"
        P1 = cv2.bitwise_and (prev_intersect, next_intersect)
        # prev_intersect_dilate = cv2.dilate (prev_intersect, kernel10)
        # next_intersect_dilate = cv2.dilate (next_intersect, kernel10)
        prev_intersect_dilate = process_morphological (prev_intersect) 
        next_intersect_dilate = process_morphological (next_intersect)
        
        P2 = cv2.bitwise_and (prev_intersect_dilate, next_intersect_dilate)
        fg_3frame = cv2.bitwise_xor (P1, P2)

        #? select fg method
        # 1. combine 3frame and MOG
        # fg = cv2.bitwise_xor (fg_mog, fg_3frame)
        # 2. MOG only
        # fg = fg_mog
        # 3. 3frame modified only
        # fg = fg_3frame
        # 4. P2
        fg = P2
        # 5. Orig 3frame
        # fg = P1

        # remove noise
        fg = process_morphological (fg)
        # apply mask
        fg = cv2.bitwise_and (fg, masks[_id][view])

        # get blobs
        blobs = get_contours (fg)

        #? for drawing, choose you want :
        #1. prev img
        # frame = prev_img[_id][view][1]
        #2. fg
        # fg_color = cv2.cvtColor (fg, cv2.COLOR_GRAY2BGR) 
        # frame = fg_color
        #3. prev img color
        frame = prev_img_color[_id][view][1]

        #?. whether drawing bounding box
        # frame = draw_bounding_box_contours (fg_color, blobs)

        # max angle from vp2 for this frame only
        max_angle_vp2 = None
        max_point_vp2 = None
        min_angle_vp2 = None
        min_point_vp2 = None

        for b in blobs : 
            # first get from left to right
            c1_right, c1_left = get_extreme_tan_point (vp1, b)
            c2_right, c2_left = get_extreme_tan_point (vp2, b)
            c3_right, c3_left = get_extreme_tan_point (None, b)

            # get helper line
            line = [
                    Line.from_two_points (vp1, c1_left),
                    Line.from_two_points (vp2, c2_right),
                    Line (None, c3_left[0])
                ]

            # draw helper line, 
            # for l_idx, l in enumerate (line[:-1]) : 
            #     color = [0, 0, 0]
            #     color[l_idx] += 255
            #     frame = l.draw (frame)

            # get corner point 
            cp = [
                    line[0].get_intersection (line[1]),
                    line[1].get_intersection (line[2])
                ]
            # draw corner point
            # for _ in cp : 
            #     frame = cv2.circle (frame, tuple ([int (__) for __ in _]), 10, (0,0,255), -1)

            # max-min for right-left
            for c_idx, c in enumerate (cp) : 
                # compute angle
                ang = math.atan2 (c[1] - vp1[1], c[0] - vp1[0])

                # compare with  global  value
                if max_angle_vp1[view_idx] is None or max_angle_vp1[view_idx] < ang : 
                    max_angle_vp1[view_idx] = ang
                    max_point_vp1[view_idx] = c

                if min_angle_vp1[view_idx] is None or min_angle_vp1[view_idx] > ang : 
                    min_angle_vp1[view_idx] = ang
                    min_point_vp1[view_idx] = c

                # draw - add to corner_mask
                # corner_mask[view_idx] = cv2.circle (corner_mask[view_idx], tuple ([int (_) for _ in c]), 10, (255,255,255), -1)

            # we use again corner point detected, but since
            # all cp will be have same angle toward vp2, then we just use 
            # the last cp (c)
            # compute angle
            ang = math.atan2 (c[1] - vp2[1], c[0] - vp2[0])

            if min_angle_vp2 is None or min_angle_vp2 > ang : 
                min_angle_vp2 = ang
                min_point_vp2 = c

        # convert max top-bottom from point to polar line
        if min_point_vp2 is not None : 
            # first find its line parameter
            # just take the furthest (bottom)
            l = Line.from_two_points (vp2, min_point_vp2)
            line_extreme_vp2.append (l)
        else : 
            line_extreme_vp2.append (None)


        # draw line of extreme left-right corner point
        if max_point_vp1[view_idx] is not None : 
            max_line = Line.from_two_points (vp1, max_point_vp1[view_idx])
            min_line = Line.from_two_points (vp1, min_point_vp1[view_idx])

            frame = max_line.draw (frame, size=5, color=(0, 0, 255))
            frame = min_line.draw (frame, size=5, color=(0, 0, 255))

        # frame = topbot_mask[view_idx]
        # frame = fg 

        # combine each view
        if view_frame is None :
            view_frame = frame
        else :
            view_frame = np.hstack ((frame, view_frame))

        prev_frame[view_idx] = frame.copy ()

        # update frame difference
        prev_img[_id][view][0] = prev_img[_id][view][1]
        prev_img[_id][view][1] = img

        prev_img_color[_id][view][0] = prev_img_color[_id][view][1]
        prev_img_color[_id][view][1] = img_color

    frame = view_frame

    if not (TBL.is_found0 and TBL.is_found1) : 
        TBL.update (line_extreme_vp2)
    else : 
        # assumption the right and left is already stable
        if ctr == 500 : 
            print (TBL.bucket_max)
            line_result = []
            for t in TBL.bucket_max : 
                for i, v in enumerate (vps[_id]) : 
                    this_bin = (t[0][2*i]+1, t[0][2*i + 1]+1)
                    line_result.append (TBL.inverse_bin_Line (this_bin, vps[_id][v]['vp2']))

            # draw top bottom
            for l_idx, l in enumerate (line_result) : 
                prev_frame[l_idx % 3] = l.draw (prev_frame[l_idx % 3], color=(0,0,255))

            # draw ground truth

            for i, view in enumerate (session[_id]) : 
                # draw line
                lines =  [ 
                        Line.from_two_points (common_plane_coor['session0'][view][0], vps[_id][view]['vp2']),  # top
                        Line.from_two_points (common_plane_coor['session0'][view][-1], vps[_id][view]['vp2']), # bottom
                        Line.from_two_points (common_plane_coor['session0'][view][1], vps[_id][view]['vp1']), # right 
                        Line.from_two_points (common_plane_coor['session0'][view][2], vps[_id][view]['vp1']), # left 
                    ]

                for l in lines : 
                    prev_frame[i] = l.draw (prev_frame[i], color=(0, 255, 0))
                cv2.imwrite ('result/bottop-{}.jpg'.format (i), prev_frame[i])
            sys.exit ()


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
