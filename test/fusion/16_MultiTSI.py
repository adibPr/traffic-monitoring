#!/usr/bin/env python2
from __future__ import print_function, division
import os
import sys
import json
import time
import random
import logging
import shutil

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

logger.info ("Load ground truth")
GT_OCC = np.load ('ground_truth/gt_all.npy')
GT_LIST_ID = [0,1,2,4,5,6,8,13,14,15,16,17,18,19,21,24,25,26,27,28,29,31,32,34,35,36,37,38,40,41,42,43,44,45,46,47,51]
CLIP_ID = [0, 1, 13, 18, 26, 31]
CLIP_IDX = [ idx  for (idx,_id) in enumerate (GT_LIST_ID) if _id in CLIP_ID ] # list index of selected ID

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

# for saving result
# detected_vehicle = {}
# for view in VIEW : 
#     detected_vehicle[view] = np.zeros (GT_OCC.shape)
# detected_vehicle['combine'] = np.zeros (GT_OCC.shape)
# detected_vehicle['PFM'] = np.zeros (GT_OCC.shape)

for tot_strip in range (10, 210, 10) : 
    logger.info ("Start tot_strip = {}".format (tot_strip))
    for case in data_path :
        logger.info ("Start case {}".format (case))

        data_id, ses_id, _ = case.split ('-')
        data_id = int (data_id)
        data_idx = GT_LIST_ID.index (data_id)
        ses_id = int (ses_id)
        if data_id != 0 : 
            continue

        logger.info ("Get VP")
        VP = VPLoader ()
        vp = VP.get_session (ses_id)

        fi = {} # from frame iterator
        M = {} # matrix homography
        M_inv = {} # matrix homography
        masks = {}

        # tsi
        # tsi_object = {}
        fdiff_tsi = {}
        tsi_const = {}
        tsi_const['total'] = tot_strip 
        tsi_const['xrange'] = int (1000 / tsi_const['total'])
        fdiff_tsi_multi = {}

        # for view
        fdiff_view = {}

        # miscelanous
        empty = np.zeros ((300, 1000)).astype ('uint8')
        random.seed (200)

        """
        # make if not exist
        saved_path = os.path.join (os.path.abspath (os.path.dirname (__file__)), '16_MultiTSI', 'MultiTSI', '{}-{}-{}-view'.format (data_id, ses_id, tsi_const['total']))
        if os.path.isdir (saved_path) : 
            # remove it
            shutil.rmtree (saved_path)

        logger.info ("Make directory - {}".format (saved_path))
        os.makedirs (saved_path)
        """

        for view in VIEW : 
            logger.info ("-- Start {} view".format (view))

            # frame iterator
            logger.info ("-- Create FrameIterator object")
            fi[view] = FrameIterator (os.path.join (occlusion_clip_path, case, view))
            points = GT['session{}'.format (ses_id)][view]
            corner = TSIUtil.get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points)

            # get rectangular homography mapping
            logger.info (" -- Get homography mapping")
            corner_gt = np.float32 (corner)
            corner_wrap = np.float32 ([[0,300],[0,0], [1000,0], [1000, 300]])
            M[view] = cv2.getPerspectiveTransform (corner_gt, corner_wrap)
            M_inv[view] = cv2.getPerspectiveTransform (corner_wrap, corner_gt)

            # for tsi object
            logger.info ("-- Create TSI object")
            # tsi_object[view] = TSIUtil.TSI (M[view])

            # for 3 frame difference
            # prev_tsi = [None, None]
            prev_view = [None, None]
            prev_tsi_multi = [None, None]
            
            logger.info ("-- Generate prev_epi and tsi")
            for i in range (2) : 
                img_color = img = next (fi[view])
                img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

                tsi_multi = None
                dst = cv2.warpPerspective (img, M[view], (1000, 300))
                new_dst = dst.copy ()
                for j in range (tsi_const['total']): 
                    strip = dst[:, j*tsi_const['xrange']:j*tsi_const['xrange']+3]
                    if tsi_multi is None : 
                        tsi_multi = strip
                    else : 
                        tsi_multi = np.hstack ((tsi_multi, strip))
                    cv2.line (new_dst, (j*tsi_const['xrange'], 0), (j*tsi_const['xrange'], 300), color=(255,255,0), thickness=3)

                # prev_tsi[i] = tsi_object[view].apply (dst)
                prev_view[i] = dst
                prev_tsi_multi[i] = tsi_multi

            logger.info ("-- Initialize TSI FrameDifference")
            # fdiff_tsi[view] = FrameDifference (*prev_tsi)
            # fdiff_view[view] = BackgroundModel (iter (prev_view), detectShadows=False)
            # fdiff_view[view].learn (tot_frame_init=2)
            fdiff_view[view] = FrameDifference (*prev_view)

            # fdiff_tsi_multi[view] = BackgroundModel (iter (prev_tsi_multi), detectShadows=False)
            # fdiff_tsi_multi[view].learn (tot_frame_init=2)
            fdiff_tsi_multi[view] = FrameDifference (*prev_tsi_multi)

            mask_path = '../../data/gt/2016-ITS-BrnoCompSpeed/dataset/session{}_{}/video_mask.png'.format (ses_id, view)
            masks[view] = cv2.imread (mask_path, 0)

        # flag_prev_object = False
        # list_height = []
        # logger.info ("Creating tsi_multi")

        logger.info ("Done initializing")
        logger.info ("Start counting")

        """
        -------------------------------------------------------------------------------
        Main Program
        -------------------------------------------------------------------------------
        """

        try : 
            ctr = -1
            while True:
                ctr += 1
                logger.info ("Iteration - {}".format (ctr))

                intersection_tsi_multi = None
                intersection_view = None
                disp = None
                t_begin = time.time ()

                tsi_multi_color = {}

                logger.info ("Processing per view")
                dsts = {}

                for view in VIEW : 
                    img = next (fi[view])
                    # img = cv2.cvtColor (img_color, cv2.COLOR_BGR2GRAY)

                    tsi_multi = None

                    dst = cv2.warpPerspective (img, M[view], (1000, 300))
                    dsts[view] = dst
                    
                    # """
                    logger.info (" Generating TSI Multi")
                    for j in range (tsi_const['total']): 
                        strip = dst[:, j*tsi_const['xrange']:j*tsi_const['xrange']+3].copy ()
                        if tsi_multi is None : 
                            tsi_multi = strip
                        else : 
                            tsi_multi = np.hstack ((tsi_multi, strip))
                    logger.info (".. Done")

                    tsi_multi_color[view] = tsi_multi
                    tsi_multi = cv2.cvtColor ( tsi_multi, cv2.COLOR_BGR2GRAY )

                    logger.info ("Extracting foreground MultiTSI")
                    fg_tsi_multi = fdiff_tsi_multi[view].apply (tsi_multi)
                    fg_tsi_multi = process_morphological (fg_tsi_multi, iterations=1)
                    logger.info (".. Done")


                    # drawing MultiTSI
                    blobs = get_contours (fg_tsi_multi, min_width=20, min_area=100)
                    tot_blobs = len (blobs)
                    blobs_color = cv2.cvtColor (fg_tsi_multi, cv2.COLOR_GRAY2BGR) # convert into BGR
                    blobs_color = draw_bounding_box_contours (blobs_color, blobs) # so we can draw bounding box
                    loc = (10, blobs_color.shape[0] - 5)
                    cv2.putText (blobs_color, '{}.{} #{}'.format (view[0], ctr, tot_blobs), loc, cv2.FONT_HERSHEY_PLAIN, 1, (0, 128, 128), 1) # and so we can insert text

                    tsi_multi = tsi_multi_color[view]
                    # """

                    """
                    logger.info (" Extracting foreground view")

                    dst = cv2.cvtColor ( dst, cv2.COLOR_BGR2GRAY )
                    fg_view = fdiff_view[view].apply (dst)
                    fg_view = process_morphological (fg_view, iterations=1)

                    logger.info (".. Done")

                    blobs = get_contours (fg_view, min_width=20, min_area=100)
                    tot_blobs = len (blobs)
                    blobs_color = cv2.cvtColor (fg_view, cv2.COLOR_GRAY2BGR) # convert into BGR
                    blobs_color = draw_bounding_box_contours (blobs_color, blobs) # so we can draw bounding box
                    loc = (10, blobs_color.shape[0] - 5)
                    cv2.putText (blobs_color, '{}.{} #{}'.format (view[0], ctr, tot_blobs), loc, cv2.FONT_HERSHEY_PLAIN, 1, (0, 128, 128), 1) # and so we can insert text
                    # """


                    if disp is None : 
                        # """ For MultiTSI
                        disp =   np.hstack (( tsi_multi, blobs_color ))
                        intersection_tsi_multi = fg_tsi_multi
                        # """

                        """ For View
                        disp = np.vstack (( dsts[view], blobs_color )) 
                        intersection_view = fg_view
                        # """
                    else : 
                        # """ For MultiTSI
                        disp = np.hstack ((disp, tsi_multi, blobs_color ))
                        intersection_tsi_multi = cv2.bitwise_and (intersection_tsi_multi, fg_tsi_multi)
                        # """

                        """ For View
                        disp = np.vstack (( disp, dsts[view], blobs_color )) 
                        intersection_view = cv2.bitwise_and (intersection_view, fg_view)
                        # """

                t_end = time.time ()
                logger.info ("Done in {:.5F}s ".format (t_begin-t_end))
                print ("{}, {:.5F}s ".format (data_id, t_end-t_begin))
                break

                # """ Coloring combined tsiMulti
                blobs = get_contours (intersection_tsi_multi, min_width=20, min_area=100)
                tot_blobs = len (blobs)

                intersection_tsi_multi = cv2.cvtColor (intersection_tsi_multi, cv2.COLOR_GRAY2BGR)
                intersection_tsi_multi = draw_bounding_box_contours (intersection_tsi_multi, blobs)
                loc = (10, intersection_tsi_multi.shape[0] - 5)
                cv2.putText (intersection_tsi_multi, '*.{} #{}'.format (ctr, tot_blobs), loc, cv2.FONT_HERSHEY_PLAIN, 1, (0, 128, 128), 1) # and so we can insert text
                # """

                """ Coloring combined view
                blobs = get_contours (intersection_view, min_width=20, min_area=100) 
                tot_blobs = len (blobs)

                intersection_view = cv2.cvtColor (intersection_view, cv2.COLOR_GRAY2BGR)
                intersection_view = draw_bounding_box_contours (intersection_view, blobs)
                loc = (10, intersection_view.shape[0] - 5)
                cv2.putText (intersection_view, '*.{} #{}'.format (ctr, tot_blobs), loc, cv2.FONT_HERSHEY_PLAIN, 1, (0, 128, 128), 1) # and so we can insert text
                # """

                ## display
                # disp = np.vstack (( disp, intersection_view ))
                disp = np.hstack (( disp, intersection_tsi_multi ))
                cv2.imshow ('default', disp)
                # cv2.imwrite ('16_MultiTSI/MultiTSI/{}-{}-{}-view/{}.jpg'.format (data_id, ses_id, tsi_const['total'], str (ctr).zfill (4)), disp)

                if (cv2.waitKey(1) & 0xFF == ord('q')) :
                    break

        except KeyboardInterrupt as e : 
            sys.exit ()
        except StopIteration as e : 
            continue

# for (key, value) in detected_vehicle.items () : 
#     logger.info ("Saving - {}".format (key))
#     np.save ('ground_truth/{}.npy'.format (key), value)
