#!/usr/bin/env python2
#!/usr/bin/env python

from __future__ import print_function, division
import os
import sys
import json

import cv2
import numpy as np

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..'))

from geometry import get_extreme_tan_point, Line, get_extreme_side_point, find_right_most_point
from util import *
from iterator import FrameIterator

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

def draw_polylines (img, corner) : 
    img = img.copy ()

    cornerInt = []
    for c in corner : 
        cornerInt.append (tuple ([int (_) for _ in c]))

    corner = np.array (corner, np.int32)
    corner = corner.reshape ((-1, 1, 2))

    cv2.polylines (img, [corner], True ,color=(0,0,255), thickness=5)
    return img

VP = VPLoader ()

VIEW = ("right", "center", "left")
with open ("/home/adib/My Git/traffic-monitoring/data/sync_25fps/common_point/result.json", "r") as f_buf : 
    GT = json.load (f_buf)

cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)

for ses_id in range (0, 7) : 
    if ses_id == 4 : 
        continue

    # some variable
    vp = VP.get_session (ses_id)
    fi = FrameIteratorLoader.get_session (ses_id)
    for view in VIEW : 
        img = next (fi[view])

        points = GT['session{}'.format (ses_id)][view]
        corner = get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points)

        img = draw_polylines (img, corner)

        cv2.imwrite ('result/common_ground_gt/{}-{}.jpg'.format (
            ses_id,
            view
        ), img )


