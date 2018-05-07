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
prev_fg = {_id : {}}
for view in session[_id] : 
    prev_fg[_id][view] = [None, None]
    for i in range (2) : 
        img = next (fi[_id][view])

        # by background subtraction
        fg = bms[_id][view].apply (img)
        # remove shadows, i.e value 127
        fg = cv2.threshold (fg, 200, 255, cv2.THRESH_BINARY)[1]

        # remove noise
        fg = process_morphological (fg)
        # apply mask
        fg = cv2.bitwise_and (fg, masks[_id][view])
        
        # save background
        prev_fg[_id][view][i] = fg 

# for storing extrema of vp1
max_angle_vp1 = [None, None, None]
max_point_vp1 = [None, None, None]
min_angle_vp1 = [None, None, None]
min_point_vp1 = [None, None, None]

corner_mask = [np.zeros (fg.shape), np.zeros (fg.shape), np.zeros (fg.shape)]

# constant for bucketing
TOT_BUCKET_D = 200
TOT_BUCKET_THETA = 60 # per circle, so from -2pi to 2pi will be theta*2 bucket
RATIO_BUCKET_D = math.sqrt (img.shape[1] ** 2 + img.shape[0] ** 2) / TOT_BUCKET_D
RATIO_BUCKET_THETA = float (4) / (TOT_BUCKET_THETA * 2)

bucket_freq = [{}, {}]
bucket_max = [(None, -1), (None, -1)]

ctr = 0
while True :
    view_frame = None
    prev_frame = [None, None, None]

    line_params_top = []
    line_params_bot = []
    final_params_combine = []

    # load image from each view
    for view_idx, view in enumerate (session[_id]) :

        img = next (fi[_id][view])

        vp1 = vps[_id][view]['vp1']
        vp2 = vps[_id][view]['vp2']

        # by background subtraction
        fg = bms[_id][view].apply (img)
        # remove shadows, i.e value 127
        fg = cv2.threshold (fg, 200, 255, cv2.THRESH_BINARY)[1]

        # remove noise
        fg = process_morphological (fg)
        # apply mask
        fg = cv2.bitwise_and (fg, masks[_id][view])

        # combine info from prev_fg
        # prev_fg_intersect = cv2.bitwise_or (prev_fg[_id][view][1], prev_fg[_id][view][0])
        # curr_fg_intersect = cv2.bitwise_or (fg, prev_fg[_id][view][1])
        # fgs = cv2.bitwise_and (prev_fg_intersect, curr_fg_intersect)


        # get blobs
        blobs = get_contours (fg)

        # drawing
        frame = img
        # frame = draw_bounding_box_contours (frame, blobs)
        # frame = fg
        # frame = cv2.cvtColor (fg, cv2.COLOR_GRAY2BGR)

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

            # draw line
            for l_idx, l in enumerate (line) : 
                color = [0, 0, 0]
                color[l_idx] += 255
                frame = l.draw (frame, color=color)

            # get corner point 
            cp = [
                    line[0].get_intersection (line[1]),
                    line[1].get_intersection (line[2])
                ]

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

            # compare with current frame
            if max_angle_vp2 is None or max_angle_vp2 < ang : 
                max_angle_vp2 = ang
                max_point_vp2 = c

            if min_angle_vp2 is None or min_angle_vp2 > ang : 
                min_angle_vp2 = ang
                min_point_vp2 = c

        # convert max top-bottom from point to polar line
        if max_point_vp2 is not None : 
            # first find its line parameter
            line_params = [
                    Line.from_two_points (vp2, max_point_vp2),
                    Line.from_two_points (vp2, min_point_vp2)
                ]

            # then convert to polar
            polars = []
            for lp_idx, lp in enumerate (line_params) : 
                # check first, wether the intersection with y axis still
                # in frame
                if lp.b > frame.shape[0] : 
                    polars.append (None)
                else : 
                    # draw top-bottom line
                    frame = lp.draw (frame, size=5, color=(0, 0, 255))
                    polars.append (lp.to_polar ())

            # convert it in its index form
            # both for top and bottom
            if polars[0] is not None : 
                line_params_top.extend ((
                    int (polars[0][0] / RATIO_BUCKET_D), 
                    int ((polars[0][1] + 2) / RATIO_BUCKET_THETA)# +2 to make all positive
                )) 
            else : 
                line_params_top.extend ((None, None))

            if polars[1] is not None :
                line_params_bot.extend ((
                    int (polars[1][0] / RATIO_BUCKET_D), 
                    int ((polars[1][1] + 2) / RATIO_BUCKET_THETA)
                )) 
            else : 
                line_params_bot.extend ((None, None))

        # draw line of extreme left-right corner point
        if max_point_vp1[view_idx] is not None : 
            max_line = Line.from_two_points (vp1, max_point_vp1[view_idx])
            min_line = Line.from_two_points (vp1, min_point_vp1[view_idx])

            frame = max_line.draw (frame, size=5, color=(200, 139, 219))
            frame = min_line.draw (frame, size=5, color=(200, 139, 219))

        # combine each view
        if view_frame is None :
            view_frame = frame
        else :
            view_frame = np.hstack ((frame, view_frame))

        prev_frame[view_idx] = frame.copy ()

    frame = view_frame

    # convert into tuple, so it can hashable
    line_params_top = tuple (line_params_top)
    line_params_bot = tuple (line_params_bot)

    # only consider that complete and no None exist
    # for top
    if len (line_params_top) == 6 and all ([_ is not None for _ in line_params_top]) : 
        bucket_freq[0][line_params_top] = bucket_freq[0].get (line_params_top, 0) + 1

        if bucket_freq[0][line_params_top] > bucket_max[0][1] :
            bucket_max[0] = (line_params_top, bucket_freq[0][line_params_top])

    # and for bottom
    if len (line_params_bot) == 6 and all ([_ is not None for _ in line_params_bot]) : 
        bucket_freq[1][line_params_bot] = bucket_freq[1].get (line_params_bot, 0) + 1

        if bucket_freq[1][line_params_bot] > bucket_max[1][1] :
            bucket_max[1] = (line_params_bot, bucket_freq[1][line_params_bot])

    # print ("T : {} : {}".format (line_params_top, bucket_freq[0].get (line_params_top, 0)))
    # print ("B : {} : {}".format (line_params_bot, bucket_freq[1].get (line_params_bot, 0)))

    # if found any index that occurance greater than threshold, exit
    if all ([_[1] >= 5 for _ in bucket_max]) : 
        print (bucket_max)
        sys.exit ()


    # drawing
    # put text of iterator
    loc = (20, frame.shape[0]-20)
    cv2.putText (frame, 'Frame - {}'.format (ctr + 1), loc, cv2.FONT_HERSHEY_PLAIN, 3, (0, 128, 128), 4)

    # show image
    cv2.imshow ('default', frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')) :
        break

    if ctr == 10 :
        """
        # also wirte  upper an bottom 
        with open ('result.json','r') as f_buff : 
            lines = json.load (f_buff)['result']
            for line in lines[:1] : 
                line = line[0]
                for i in range (3): 
                    r, theta = line[i*2], line[i*2+1]
                    r *= RATIO_BUCKET_D
                    theta = (theta * RATIO_BUCKET_THETA) - 2
                    print (r, theta)

                    # draw r, theta by convert it into a,b first
                    # l = Line.from_polar (r, theta)
                    x = r * math.cos (theta)
                    y = r * math.sin (theta)
                    l = Line.from_two_points ((x,y), vp2)
                    prev_frame[i] = l.draw (prev_frame[i], size=5)
        # """

        # write both max line and corner
        for i in range (3) : 
            cv2.imwrite ('line-{}.jpg'.format (i), prev_frame[i])
            cv2.imwrite ('track-{}.jpg'.format (i), corner_mask[i])
        
        # sys.exit ()
    
    ctr += 1
    print (ctr)
