#!/usr/bin/env python2
from __future__ import print_function, division
import os
import sys
import json
import time
import random
import logging

import cv2
import numpy as np

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..', '..'))

import geometry 
from util import *
from iterator import FrameIterator
from background import BackgroundModel, FrameDifference
import TSIUtil

"""
-------------------------------------------------------------------------------
Some Inititalization
-------------------------------------------------------------------------------
"""
logging.basicConfig (level=logging.INFO)
logger = logging.getLogger ("[main]")

logger.info ("Start program")
logger.info ("Start Initializ")
TOT_LANE = [3, 2, 2, 2, 2, 2, 3]  
CLIP_ID = [0, 1, 13, 18, 26, 31]
EPI_LANE_IDX = [[75, 150, 225], [50, 100, 150, 200, 250]]

logger.info ("Load ground truth")
VIEW = ("right", "center", "left")
with open ("/home/adib/My Git/traffic-monitoring/data/sync_25fps/common_point/result.json", "r") as f_buf : 
    GT = json.load (f_buf)

cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)
img_path = '/home/adib/My Git/traffic-monitoring/data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/screen.png'
occlusion_clip_path = '/home/adib/My Git/traffic-monitoring/data/occlusion/'

logger.info ("Filter data clips")
data_path = [ d \
        for d in os.listdir (occlusion_clip_path) \
            if d != 'all_video' \
            and os.path.isdir (os.path.join (occlusion_clip_path, d)) \
            and int (d.split('-')[0]) in CLIP_ID 
        ]
data_path = sorted (data_path, key = lambda p: int (p.split ('-')[0]))
detected_vehicle_all = None 

for case in data_path :
    logger.info ("Start case {}".format (case))

    _id, ses_id, _ = case.split ('-')
    ses_id = int (ses_id)
    tot_lane = TOT_LANE[ses_id]

    logger.info ("Get VP")
    VP = VPLoader ()
    vp = VP.get_session (ses_id)

    fi = {} # from frame iterator
    M = {} # matrix homography
    M_inv = {} # matrix homography
    imgs_color = {} # for saving image color each view
    masks = {}
    fgs = {}
    prev_imgs_color = {}

    # epi
    tot_epi = tot_lane * 2 - 1
    epi_line_idx = EPI_LANE_IDX[tot_lane-2] 
    epi_object = {}
    fdiff_epi = {}
    logger.info ("- Tot epi : {}, idx : {}".format (tot_epi, " ".join ([str (s) for s in epi_line_idx])))

    # tsi
    tsi_object = {}
    fdiff_tsi = {}

    # miscelanous
    empty = np.zeros ((300, 1000)).astype ('uint8')
    random.seed (200)

    for view in VIEW : 
        logger.info ("-- Start {} view".format (view))

        # frame iterator
        logger.info ("-- Crate FrameIterator object")
        fi[view] = FrameIterator (os.path.join (occlusion_clip_path, case, view))
        points = GT['session{}'.format (ses_id)][view]
        corner = TSIUtil.get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points)

        # get rectangular homography mapping
        corner_gt = np.float32 (corner)
        corner_wrap = np.float32 ([[0,300],[0,0], [1000,0], [1000, 300]])
        M[view] = cv2.getPerspectiveTransform (corner_gt, corner_wrap)
        M_inv[view] = cv2.getPerspectiveTransform (corner_wrap, corner_gt)

        # for epi object
        logger.info ("-- Create {} EPI object".format (tot_epi))
        epi_object[view] = []
        for idx in epi_line_idx : 
            epi_object[view].append (TSIUtil.EPI (M[view],
                    size=(1000, 300),
                    VDL_IDX=idx, 
                    VDL_SIZE=3, 
                ))

        # for tsi object
        logger.info ("-- Create TSI object")
        tsi_object[view] = TSIUtil.TSI (M[view])

        # for 3 frame difference
        prev_epi = [[None, None] for i in range (tot_epi)]
        prev_tsi = [None, None]
        
        logger.info ("-- Generate prev_epi and tsi")
        for i in range (2) : 
            img_color = img = next (fi[view])
            prev_imgs_color[view] = img
            img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

            dst = cv2.warpPerspective (img, M[view], (1000, 300))
            for j in range (tot_epi) : 
                prev_epi[j][i] = epi_object[view][j].apply (dst)
            prev_tsi[i] = tsi_object[view].apply (dst)


        logger.info ("-- Initialize EPI FrameDifference")
        fdiff_epi[view] = [] 
        for i in range (tot_epi) : 
            fdiff_epi[view].append (FrameDifference (*prev_epi[i]))

        logger.info ("-- Initialize TSI FrameDifference")
        fdiff_tsi[view] = FrameDifference (*prev_tsi)


        mask_path = '../../data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/video_mask.png'.format (ses_id, view)
        masks[view] = cv2.imread (mask_path, 0)

    flag_prev_object = False
    list_height = []

    logger.info ("Done initializing")
    logger.info ("Start counting")

    """
    -------------------------------------------------------------------------------
    Main Program
    -------------------------------------------------------------------------------
    """
    detected_vehicle = np.zeros ((149))
    try : 
        ctr = -1
        while True:
            ctr += 1
            logger.info ("Iteration - {}".format (ctr))

            intersection_epi = [None for i in range (tot_epi)] 
            intersection_tsi = None
            disp = None
            t_begin = time.time ()

            logger.info ("Processing per view")
            epi = [ None for i in range (tot_epi) ]
            epi_fg = [ None for i in range (tot_epi) ]

            for view in VIEW : 
                img_color = next (fi[view])
                imgs_color[view] = img_color
                img = cv2.cvtColor (img_color, cv2.COLOR_BGR2GRAY)

                dst = cv2.warpPerspective (img, M[view], (1000, 300))

                for i in range (tot_epi) : 
                    epi[i] = epi_object[view][i].apply (img)
                    epi_fg[i] = fdiff_epi[view][i].apply (epi[i])
                    epi_fg[i] = process_morphological (epi_fg[i], iterations=2)

                tsi = tsi_object[view].apply (img)
                tsi_fg = fdiff_tsi[view].apply (tsi)
                tsi_fg = process_morphological (tsi_fg, iterations=2)

                if disp is None : 
                    disp =  tsi 
                    for i in range (tot_epi) : 
                        intersection_epi[i] = epi_fg[i]

                    intersection_tsi = tsi_fg
                else : 
                    disp = np.vstack ((disp, tsi))
                    for i in range (tot_epi) : 
                        intersection_epi[i] = cv2.bitwise_and (intersection_epi[i], epi_fg[i])
                    intersection_tsi = cv2.bitwise_and (intersection_tsi, tsi_fg)

            t_end = time.time ()
            logger.info ("Done in {:.5F}s ".format (t_begin-t_end))

            # drawing session
            dst_color = cv2.warpPerspective (imgs_color['center'], M['center'], (1000, 300))
            for i in range (tot_epi) : 
                cv2.line (dst_color, (0, epi_line_idx[i]-1), (1000, epi_line_idx[i] + 1), color=(255, 255, 0), thickness=2) 

            # making tsi  strip
            tsi_strip = intersection_tsi[:, 0:5]
            empty[:, 0:5] = tsi_strip

            # making epi strip and count vehicle
            this_tot = 0
            for i in range (tot_epi) : 
                epi_strip = intersection_epi[i][:15, :]
                blobs = get_contours (epi_strip, min_area=0, min_width=0)
                blobs = [ b for b in blobs if cv2.boundingRect (b)[2] >= 50 ]
                this_tot += len (blobs)

                empty[epi_line_idx[i]-7:epi_line_idx[i] + 8,:] = epi_strip

            detected_vehicle[ctr] = this_tot

            empty_color = cv2.cvtColor (empty, cv2.COLOR_GRAY2BGR)
            intersection_epi_combined = None 
            for i in range (tot_epi) : 
                ec = cv2.cvtColor (intersection_epi[i], cv2.COLOR_GRAY2BGR)
                if intersection_epi_combined is None :
                    intersection_epi_combined = ec 
                else : 
                    intersection_epi_combined = np.vstack ((intersection_epi_combined, ec))


            disp = np.vstack ((dst_color, empty_color, intersection_epi_combined))#, intersection_tsi, intersection_epi))
            loc = (10, 25)
            cv2.putText (disp, 'Tot Vehicle - {}'.format (this_tot), loc, cv2.FONT_HERSHEY_PLAIN, 3, (0, 128, 128), 4)
            cv2.imshow ('default', disp)

            if (cv2.waitKey(1) & 0xFF == ord('q')) :
                break

    except KeyboardInterrupt as e : 
        sys.exit ()
    except StopIteration as e : 
        if detected_vehicle_all is None : 
            detected_vehicle_all = detected_vehicle
        else : 
            detected_vehicle_all = np.vstack ((detected_vehicle_all, detected_vehicle))
        continue

detected_vehicle_all = detected_vehicle_all.transpose ()
np.savetxt ('result.txt', detected_vehicle_all, delimiter=',')

