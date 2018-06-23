#!/usr/bin/env python2
from __future__ import print_function, division
import os
import sys
import json
import time
import random
import logging
import shutil
import csv
from collections import defaultdict 

import cv2
import numpy as np

path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..', '..', '..'))

import geometry 
from util import *
from iterator import FrameIterator
from background import BackgroundModel, FrameDifference
import TSIUtil

class ElapsedFormatter(logging.Formatter):

    def __init__ (self, *args, **kwargs) : 
        self.prev_msg_time = time.time ()
        super (ElapsedFormatter, self).__init__ (*args, **kwargs)

    def format(self, record):
        curr_time = time.time ()
        elapsed = curr_time - self.prev_msg_time 

        res = super (ElapsedFormatter, self).format (record)
        res = "{:.2F}".format (elapsed) + "   " + res
        self.prev_msg_time = curr_time

        return res


"""
-------------------------------------------------------------------------------
Some Inititalization
-------------------------------------------------------------------------------
"""
cv2.namedWindow ('default', flags=cv2.WINDOW_NORMAL)
VIEW = ("right", "center", "left")
logging.basicConfig (level=logging.INFO)
logger = logging.getLogger ("[main]")
formatter = ElapsedFormatter (fmt='%(name)s %(levelname)-8s %(message)s',
                                  datefmt='%H:%M:%S')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.propagate = False
logger.addHandler(ch)


logger.info ("Start program")
logger.info ("Start Initializ")

logger.info ("Load ground truth")

class ColorBasedPFM (object) : 
    
    def __init__ (self, w_color = False) : 
        self.WITH_COLOR = w_color
        self.load_constant ()

    def load_constant (self) : 
        self.GT_LIST_ID = [0,1,2,4,5,6,8,13,14,15,16,17,18,19,21,24,25,26,27,28,29,31,32,34,35,36,37,38,40,41,42,43,44,45,46,47,51]
        with open ("/home/adib/My Git/traffic-monitoring/data/sync_25fps/common_point/result.json", "r") as f_buf : 
            self.GT = json.load (f_buf)

        self.occlusion_clip_path     = '/home/adib/My Git/traffic-monitoring/data/occlusion/'
        self.gt_one_frame_path       = '/home/adib/My Git/traffic-monitoring/data/occlusion/GT_one_frame.csv'

        logger.info ('Load ground truth a frame only')
        with open (self.gt_one_frame_path, 'r') as f_buf : 
            self.GT_ONE_FRAME = list (csv.DictReader (f_buf))

        logger.info ("Filter data clips")
        data_path = [ d \
                for d in os.listdir (self.occlusion_clip_path) \
                    if d != 'all_video' \
                    and os.path.isdir (os.path.join (self.occlusion_clip_path, d)) \
                    and int (d.split('-')[0]) in self.GT_LIST_ID
                ]
        self.data_path = sorted (data_path, key = lambda p: int (p.split ('-')[0]))

        self.VP = VPLoader ()

    def set_scene (self, data_id) : 
        # find the correct  path
        d_path = [ p for p in self.data_path if int (p.split ('-')[0]) == int (data_id) ]
        assert len (d_path) == 1
        d_path = d_path[0]

        # obtained full  path of the clip
        logger.info ("Start for folder {}".format (d_path))
        clip_id, ses_id, _ = d_path.split ('-')
        clip_id = int (clip_id)
        ses_id = int (ses_id)
        full_d_path = os.path.join (self.occlusion_clip_path, d_path)

        # obtained groundtruth of this clip
        logger.info ("Get ground truth")
        clip_gt = [ ent for ent in self.GT_ONE_FRAME if int (ent['data_id'].split ('-')[0]) == clip_id ][0]
        frame_end = int (clip_gt['Frame']) + 1 + 3
        frame_start = frame_end - 5

        logger.info ("Creating frame iterator")
        self.fi = self.iterator_range (full_d_path, frame_start, frame_end)
        self.M = {}
        self.M_inv = {}
        self.fdiff_view = {}

        for view in VIEW : 
            logger.info ("Creating object detection for {}".format (view))
            self.M[view], self.M_inv[view] = self.get_matrix_transform (ses_id, view)
            # self.fdiff_view[view] = self.init_MoG (self.fi[view], self.M[view])
            self.fdiff_view[view] = self.init_fdiff (self.fi[view], self.M[view])

    def process (self, imgs=None) :
        if imgs is None : 
            imgs = self.next_frame ()
        fgs = {}
        dsts = {}

        for view in VIEW : 
            logger.info ('Process {}'.format (view))
            img = self.preproces_image (imgs[view])
            dst = cv2.warpPerspective (img, self.M[view], (1000, 300))
            dsts[view] = cv2.warpPerspective (imgs[view], self.M[view], (1000, 300))

            logger.info (' -- Extract background')
            fg = self.fdiff_view[view].apply (dst)
            fg = process_morphological (fg, iterations=1)

            fgs[view] =  fg

        fgs['and'] = self.intersection (fgs.values ())
        return imgs, dsts, fgs

    def get_matrix_transform (self, ses_id, view) : 
        vp = self.VP.get_session (ses_id)
        points = self.GT['session{}'.format (ses_id)][view]

        corner = TSIUtil.get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points)
        corner_gt = np.float32 (corner)

        corner_wrap = np.float32 ([[0,300],[0,0], [1000,0], [1000, 300]])

        M = cv2.getPerspectiveTransform (corner_gt, corner_wrap)
        M_inv = cv2.getPerspectiveTransform (corner_wrap, corner_gt)

        return M, M_inv

    def init_MoG (self, fi, M) : 
        prev_view = [None, None]
        for i in range (2) : 
            img = img_color =  next (fi)
            img = self.preproces_image (img)

            dst = cv2.warpPerspective (img, M, (1000, 300))
            prev_view[i] = dst

        fdiff_view = BackgroundModel (iter (prev_view), detectShadows=False)
        fdiff_view.learn (tot_frame_init=2)

        self.shape = dst.shape

        return fdiff_view

    def init_fdiff (self, fi, M) : 
        prev_view = [None, None]
        for i in range (2) : 
            img = img_color =  next (fi)
            img = self.preproces_image (img)

            dst = cv2.warpPerspective (img, M, (1000, 300))
            prev_view[i] = dst

        fdiff_view = FrameDifference (*prev_view)

        self.shape = dst.shape

        return fdiff_view

    def next_frame (self) : 
        imgs = {}
        for view in VIEW : 
            imgs[view] = next (self.fi[view])
        return imgs

    def preproces_image (self, img) : 
        # first blur it with median blur
        img = cv2.GaussianBlur (img, (5,5), 0)
        # then check if its grayscale or not
        if not self.WITH_COLOR : 
            img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def iterator_range (path, start, end) : 
        """
        Load image from a path with three views, from [start, end)
        """
        fi = {}
        for view in ('left', 'right', 'center') : 
            fi[view] = []
            list_folder = sorted (os.listdir (os.path.join (path, view)))
            # since start and end start with 1
            start_index_folder = int (list_folder[0].split ('.')[0]) - 1 
            for i in range (start, end) : 
                img_full_path = os.path.join (path, view,  str (i+start_index_folder).zfill (5) + '.jpg')
                fi[view].append (cv2.imread (img_full_path))

            # print ('[{}.{}] - {}'.format (path, view, i + start_index_folder))

        for view in ('left', 'right', 'center') : 
            fi[view] = iter (fi[view])
        return fi

    @staticmethod
    def intersection (fgs) : 
        ist = fgs[0]
        for i in range (1, len (fgs)) : 
            ist = cv2.bitwise_and (ist, fgs[i])
        return ist

    @staticmethod
    def get_blobs_from_image (fg, mask) :
        """
        Based on https://stackoverflow.com/a/40826140
        get a new fg based on mask binary
        """
        if len (fg.shape) == 3 : 
            mask_color = cv2.cvtColor (mask, cv2.COLOR_GRAY2BGR) # convert first into BGR
        else : 
            mask_color = mask
        fg_out = cv2.subtract (mask_color, fg)
        fg_out = cv2.subtract (mask_color, fg_out)

        return fg_out

    @staticmethod
    def apply_sobel (img) : 
        if len (img.shape) == 3 : 
            img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
        laplacian = img
        laplacian = cv2.Canny (img, 100, 200)

        # laplacian = cv2.Sobel (laplacian, cv2.CV_64F, 0,1, ksize=5) #sobel X
        # laplacian = cv2.Sobel (laplacian, cv2.CV_64F, 1,0, ksize=5) #sobel Y 
        # laplacian = cv2.Laplacian (laplacian, cv2.CV_64F)
        return laplacian

    @staticmethod
    def color_quantization (img, K=8) : 
        # convert to np.float32
        Z = np.float32(img)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        return res2

    @staticmethod
    def get_border_blobs (blob) : 
        # just for single blob

        (x,y, w, h) = cv2.boundingRect (blob)

        tan_right = 0
        tan_top = 300 
        tan_bot = 0
        points = [c[0].tolist () for c in blob]
        # sorted based on x axis, the-0 index is the furthest
        points = sorted (points, reverse=True)

        limit = [350, -1]
        limit_per_col = [0, 0]
        for p in points : 
            if points[0][0]  - p[0] >= 30 : 
                break

            if p[1] < limit[0] : 
                limit[0] = p[1]
            if p[1] > limit[1] : 
                limit[1] = p[1]

        tan_bot = limit[1]
        tan_top = limit[0]

        length_col = defaultdict (lambda : [360,-1]) 
        for p in points : 
            x,y = p
            if y < length_col[x][0] : 
                length_col[x][0] = y 

            if y > length_col[x][1]  : 
                length_col[x][1] = y 

        for key in sorted (length_col.keys ()) : 
            this_rasio = min (limit[1], length_col[key][1]) - max (limit[0], length_col[key][0])
            limit_rasio = limit[1] - limit[0]
            print ('This rasio : {}, limit_rasio : {}, key : {}'.format (this_rasio, limit_rasio, key))
            if this_rasio  /  limit_rasio  >= 0.8 : 
                tan_left = key
                break
            else : 
                print ("Ayy")
        else : 
            tan_left = key
            
        tan_right = points[0][0]

        ground_point = [
                (tan_left, tan_bot),
                (tan_left, tan_top),
                (tan_right, tan_top),
                (tan_right, tan_bot)
            ]


        return ground_point


class CommonGroundPFM (ColorBasedPFM) : 

    def set_scene (self, data_id) : 
        # find the correct  path
        d_path = [ p for p in self.data_path if int (p.split ('-')[0]) == int (data_id) ]
        assert len (d_path) == 1
        d_path = d_path[0]

        # obtained full  path of the clip
        logger.info ("Start for folder {}".format (d_path))
        clip_id, ses_id, _ = d_path.split ('-')
        clip_id = int (clip_id)
        ses_id = int (ses_id)
        full_d_path = os.path.join (self.occlusion_clip_path, d_path)

        # obtained groundtruth of this clip
        logger.info ("Get ground truth")
        clip_gt = [ ent for ent in self.GT_ONE_FRAME if int (ent['data_id'].split ('-')[0]) == clip_id ][0]
        frame_end = int (clip_gt['Frame']) + 1
        frame_start = frame_end - 5

        logger.info ("Creating frame iterator")
        self.fi = self.iterator_range (full_d_path, frame_start, frame_end)
        self.fdiff_view = {}
        self.masks = {}

        for view in VIEW : 
            logger.info ("Creating object detection for {}".format (view))
            # self.fdiff_view[view] = self.init_MoG (self.fi[view], self.M[view])
            self.fdiff_view[view] = self.init_fdiff (self.fi[view])

            logger.info ("Creating Mask")
            self.masks[view] = np.zeros (self.shape).astype ('uint8')
            vp = self.VP.get_session (ses_id)
            points = self.GT['session{}'.format (ses_id)][view]

            # load from corner VP
            corner = TSIUtil.get_corner_ground (vp[view]['vp1'], vp[view]['vp2'], points)
            corner_gt = np.array (corner).astype ('int32')
            # apply polygon
            cv2.fillPoly (self.masks[view], [corner_gt], 255)

    def init_fdiff (self, fi) : 
        prev_view = [None, None]
        for i in range (2) : 
            img = img_color =  next (fi)
            img = self.preproces_image (img)

            prev_view[i] = img 

        fdiff_view = FrameDifference (*prev_view)

        self.shape = img.shape

        return fdiff_view

    def init_MoG (self, fi) : 
        prev_view = [None, None]
        for i in range (2) : 
            img = img_color =  next (fi)
            img = self.preproces_image (img)

            prev_view[i] = img 

        fdiff_view = BackgroundModel (iter (prev_view), detectShadows=False)
        fdiff_view.learn (tot_frame_init=2)

        self.shape = img.shape

        return fdiff_view

    def process (self, imgs=None) :
        if imgs is None : 
            imgs = self.next_frame ()
        fgs = {}

        for view in VIEW : 
            logger.info ('Process {}'.format (view))
            img = self.preproces_image (imgs[view])

            logger.info (' -- Extract background')
            fg = self.fdiff_view[view].apply (img)
            fg = process_morphological (fg, iterations=3)

            fg = cv2.bitwise_and (fg, self.masks[view])

            fgs[view] =  fg

        fgs['and'] = self.intersection (fgs.values ())
        return imgs, fgs

def main () : 
    CBP = ColorBasedPFM (w_color=False)
    for data_id in CBP.GT_LIST_ID : 
        CBP.set_scene (data_id)

        # skip to the end only
        imgs = {}
        while True : 
            try :
                imgs = CBP.next_frame ()
            except StopIteration as e : 
                break

        imgs, dsts, fgs = CBP.process (imgs)

        tot_blobs = {}
        for view in VIEW :
            blobs = TSIUtil.get_contours (fgs[view], min_width=50, min_area=100)
            tot_blobs[view] = len (blobs)

            # extract blobs
            logger.info ("Get contours for {}".format (view))
            # dsts[view] = CBP.get_blobs_from_image (dsts[view], fgs[view])
            # for i in range (3) : 
            #     dsts[view] = cv2.medianBlur (dsts[view], 5)
            # dsts[view] = CBP.color_quantization (dsts[view], K=15)
            # dsts[view] = CBP.apply_sobel (dsts[view])

            logger.info ("Drawing {}".format (view))
            # convert into BGR so we can draw it
            fgs[view] = cv2.cvtColor (fgs[view], cv2.COLOR_GRAY2BGR)
            for b in blobs : 
                corner = CBP.get_border_blobs (b)

                corner = np.array (corner, np.int32)
                corner = corner.reshape ((-1, 1, 2))

                cv2.polylines (fgs[view], [corner], True , color=(255, 0, 0), thickness=2)

            # draw bounding box
            # fgs[view] = draw_bounding_box_contours (fgs[view], blobs) 

            # put view name in top of it
            loc = (10, 50)
            cv2.putText (fgs[view], '{} #{}'.format (view, tot_blobs[view]), loc, cv2.FONT_HERSHEY_PLAIN, 2, (0, 100, 255), 3) # and so we can insert text

        fgs.pop ('and', None)
        fgs = np.vstack (fgs.values ())

        dsts = np.vstack (dsts.values ())

        # imgs = np.hstack (imgs.values ())

        # masks = np.hstack (CBP.masks.values ())

        # cv2.imwrite ('segmented/fdiff/{}-color.jpg'.format (data_id), dsts)
        # continue
        while True : 
            cv2.imshow ('default', fgs)

            if (cv2.waitKey(1) & 0xFF == ord('q')) :
                sys.exit ()
            elif (cv2.waitKey (1) >= 0) : 
                break

if __name__ == '__main__' : 
    main ()
