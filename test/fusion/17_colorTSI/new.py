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
import colorsys

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

def get_colors (n) : 
     
    def HSVToRGB(h, s, v):
     (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
     return (int(255*r), int(255*g), int(255*b))
     
    def getDistinctColors(n):
     huePartition = 1.0 / (n + 1)
     return (HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n))

    return list (getDistinctColors (n))


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
    
    def __init__ (self, **kwargs) : 
        self.WITH_COLOR = kwargs.get ('w_color', False)
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
            self.fdiff_view[view] = self.init_MoG (self.fi[view], self.M[view])
            # self.fdiff_view[view] = self.init_fdiff (self.fi[view], self.M[view])

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
            fg = process_morphological (fg, iterations=5)

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

    def split_blob (self, blob, thres_compact=0.9) : 

        # get surrounding of poly and hull
        epsilon = 0.01*cv2.arcLength(blob,True)
        poly = cv2.approxPolyDP(blob,epsilon,True)
        hull = cv2.convexHull(blob)

        # test weather it should be divided or not
        area_poly = cv2.contourArea (poly)
        area_hull = cv2.contourArea (hull)
        ratio_area = area_poly / area_hull
        # compactness is big, no need to split
        # if ratio_area >= thres_compact : 
        #     return None 

        # create mask poly
        color = [255] * len (self.shape)
        mask_poly = np.zeros (self.shape).astype ('uint8')
        cv2.fillPoly (mask_poly, [poly], tuple (color))
        mask_poly_color = mask_poly
        if self.WITH_COLOR : 
            mask_poly = cv2.cvtColor (mask_poly, cv2.COLOR_BGR2GRAY)
        mask_poly = cv2.threshold (mask_poly, 25, 255, cv2.THRESH_BINARY)[1]

        # create mask hull
        mask_hull = np.zeros (self.shape).astype ('uint8')
        cv2.fillPoly (mask_hull, [hull], tuple (color))
        mask_hull_color = mask_hull
        if self.WITH_COLOR : 
            mask_hull = cv2.cvtColor (mask_hull, cv2.COLOR_BGR2GRAY)
        mask_hull = cv2.threshold (mask_hull, 25, 255, cv2.THRESH_BINARY)[1]

        # blob diff 
        blob_diff = cv2.bitwise_xor (mask_hull, mask_poly)
        im2, contours, hierarchy = cv2.findContours(
                blob_diff, 
                cv2.RETR_TREE, 
                cv2.CHAIN_APPROX_SIMPLE
            )
        
        # get max contour
        max_cnt = contours[0] 
        max_area = cv2.contourArea (max_cnt) 
        for cnt in contours[1:] : 
            area = cv2.contourArea (cnt)
            if max_area <= area : 
                max_area = area
                max_cnt = cnt

        mask_diff = np.zeros (self.shape).astype ('uint8')
        cv2.fillPoly (mask_diff, [max_cnt], (120, 120, 120))
        max_cnt = cv2.convexHull (max_cnt)

        cutting_point = None
        cutting_point2 = None
        most_inner_dist = None

        M = cv2.moments (hull)
        cx = int (M['m10'] / M['m00'])
        cy = int (M['m01'] / M['m00'])
        centroid = np.array ((cx, cy))

        for mc in max_cnt : 
            mc = np.array (mc[0])

            ext_dist = None 
            ext_point = None
            # for p in hull : 
            #     p = np.array (p[0])

            #     # dist =  np.linalg.norm (mc-p)
            #     dist = ((mc - p) ** 2).sum ()
            #     if ext_dist is None or dist < ext_dist :
            #         ext_dist = dist
            #         ext_point = p

            dist = np.linalg.norm (mc-centroid)

            if cutting_point is None or dist < most_inner_dist : 
                cutting_point = mc
                # cutting_point2 = ext_point
                most_inner_dist = dist 

        # cv2.line (mask_diff, tuple (cutting_point), tuple (cutting_point2), (255, 255, 0), 10)
        # cv2.circle (mask_diff, tuple (cutting_point),4, (255, 255, 0), 10)
        # cv2.circle (mask_diff, (cx,cy),4, (255, 0, 0), 10)

        mask_line = np.zeros (self.shape).astype ('uint8')

        # split horizontally
        up, bot = mask_poly[:cutting_point[1], :], mask_poly[cutting_point[1]:, :]
        im2, up, hierarchy = cv2.findContours(
                up, 
                cv2.RETR_TREE, 
                cv2.CHAIN_APPROX_SIMPLE
            )
        im2, bot, hierarchy = cv2.findContours(
                bot, 
                cv2.RETR_TREE, 
                cv2.CHAIN_APPROX_SIMPLE
            )
        up_area = max ([cv2.contourArea (c) for c in up])
        bot_area = max ([cv2.contourArea (c) for c in bot])
        horizontal_diff = abs (up_area - bot_area)

        # split vertically
        left, right = mask_poly[:, :cutting_point[0]], mask_poly[:, cutting_point[0]:]
        im2, left, hierarchy = cv2.findContours(
                left, 
                cv2.RETR_TREE, 
                cv2.CHAIN_APPROX_SIMPLE
            )
        im2, right, hierarchy = cv2.findContours(
                right, 
                cv2.RETR_TREE, 
                cv2.CHAIN_APPROX_SIMPLE
            )

        left_area = max ([cv2.contourArea (c) for c in left])
        right_area = max ([cv2.contourArea (c) for c in right])
        vertical_diff = abs (left_area - right_area)

        if horizontal_diff < vertical_diff : 
            p1 = (0, cutting_point[1])
            p2 = (mask_poly.shape[1], cutting_point[1])
            cv2.line (mask_poly, tuple (p1), tuple (p2), (0, 0, 0), 3)
            print ("Should split horizontally")
        else : 
            p1 = (cutting_point[0], 0)
            p2 = (cutting_point[0], mask_poly.shape[0])
            cv2.line (mask_poly, tuple (p1), tuple (p2), (0, 0, 0), 3)
            print ("Should split vertically")

        if self.WITH_COLOR : 
            mask_line = cv2.cvtColor (mask_line, cv2.COLOR_BGR2GRAY)
        # mask_poly = cv2.bitwise_xor (mask_poly, mask_line)
        mask_poly = cv2.cvtColor (mask_poly, cv2.COLOR_GRAY2BGR)

        return  mask_poly

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
        img_orig_shape = img.shape

        # change color space
        # img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
        # img = img[:,:, :2]

        # reshape image
        Z = img.reshape ((-1, img.shape[2]))
        # Z = img.reshape ((-1, 1))
        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        # first we need center with smalles value
        am = np.argmin (np.sum (center, axis=1))

        # center = np.uint8(center)
        center = get_colors (center.shape[0])
        center[am] = (0,0, 0)
        center = np.uint8 (center)
        res = center[label.flatten()]
        res2 = res.reshape((img_orig_shape))

        return res2

    @staticmethod
    def get_border_blobs (blob, thres_init = 50, thres_dist=20) : 
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
            if points[0][0]  - p[0] >= thres_init : 
                break

            if p[1] < limit[0] : 
                limit[0] = p[1]
            if p[1] > limit[1] : 
                limit[1] = p[1]

        tan_bot = limit[1]
        tan_top = limit[0]

        length_col = defaultdict (lambda : [None,None]) 
        for p in points : 
            x,y = p

            # top limit
            if length_col[x][0] is None or y < length_col[x][0]  : 
                length_col[x][0] = y 

            # bottom limit
            if length_col[x][1] is None or y > length_col[x][1] : 
                length_col[x][1] = y 


        for key in sorted (length_col.keys (), reverse = True) : 
            lc = length_col[key]
            if (lc[1] == lc[0]) or\
                    lc[1] <= (limit[1] + limit[0]) / 2 or\
                    abs (key - points[0][0]) <= thres_init : 
                continue
            if limit[1] - length_col[key][1]  >= thres_dist : #or \
                tan_left = key
                break
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



class MultiTSIPFM (ColorBasedPFM) : 

    def __init__ (self, **kwargs) : 
        super (MultiTSIPFM, self).__init__ (**kwargs)
        self.tsi_tot_strip = kwargs.get ('tsi_tot_strip', 50)
        self.tsi_xrange = int (1000 / self.tsi_tot_strip)

    def preproces_image (self, img) : 
        # first blur it with median blur
        img = cv2.GaussianBlur (img, (5,5), 0)
        # then check if its grayscale or not
        if not self.WITH_COLOR : 
            img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
        return img

    def create_tsi (self, img) : 
        tsi_multi = None
        for j in range (self.tsi_tot_strip): 
            if len (img.shape) == 3 : 
                strip = img[:, j*self.tsi_xrange:j*self.tsi_xrange+3, :]
            else : 
                strip = img[:, j*self.tsi_xrange:j*self.tsi_xrange+3]
            if tsi_multi is None : 
                tsi_multi = strip
            else : 
                tsi_multi = np.hstack ((tsi_multi, strip))
        return tsi_multi

    def init_MoG (self, fi, M) : 
        prev_view = [None, None]
        for i in range (2) : 
            img = img_color =  next (fi)
            img = self.preproces_image (img)

            dst = cv2.warpPerspective (img, M, (1000, 300))
            dst = self.create_tsi (dst)
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
            dst = self.create_tsi (dst)
            prev_view[i] = dst

        fdiff_view = FrameDifference (*prev_view)

        self.shape = dst.shape

        return fdiff_view

    def process (self, imgs=None) :
        if imgs is None : 
            imgs = self.next_frame ()
        fgs = {}
        dsts = {}

        for view in VIEW : 
            logger.info ('Process {}'.format (view))
            img = self.preproces_image (imgs[view])
            dst = cv2.warpPerspective (img, self.M[view], (1000, 300))
            dst = self.create_tsi (dst)
            dsts[view] = dst

            logger.info (' -- Extract background')
            fg = self.fdiff_view[view].apply (dst)
            fg = process_morphological (fg, iterations=3)

            fgs[view] =  fg

        fgs['and'] = self.intersection (fgs.values ())
        return imgs, dsts, fgs

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
    CBP = MultiTSIPFM (w_color=True)
    # CBP = ColorBasedPFM (w_color=False)

    # csv_out = csv.DictWriter (open ('result.csv', 'wb'), ('data_id', 'left', 'center', 'right', 'and'))
    # csv_out.writeheader ()

    accuracy = defaultdict (lambda : 0) 
    for data_id in CBP.GT_LIST_ID[:] : 
        
        row = {}
        row['data_id'] = data_id

        CBP.set_scene (data_id)
        gt = [ ent for ent in CBP.GT_ONE_FRAME if int (ent['data_id'].split ('-')[0]) == data_id ][0]

        # skip to the end only
        imgs = {}
        while True : 
            try :
                imgs, dsts, fgs = CBP.process ()
            except StopIteration as e : 
                break

        tot_blobs = {}
        mask_combine = {}
        for view in fgs.keys () :
            blobs = TSIUtil.get_contours (fgs[view], min_width=50, min_area=100)
            tot_blobs[view] = len (blobs)

            row[view] = tot_blobs[view]

            # calculate accuracy
            if int (gt['GT']) == len (blobs) : 
                accuracy[view] += 1
            else : 
                accuracy[view] += 0

            # extract blobs
            # logger.info ("Get contours for {}".format (view))
            # dsts[view] = CBP.get_blobs_from_image (dsts[view], fgs[view])
            # for i in range (3) : 
            #     dsts[view] = cv2.medianBlur (dsts[view], 5)
            # dsts[view] = CBP.color_quantization (dsts[view], K=5)
            # dsts[view] = CBP.apply_sobel (dsts[view])

            logger.info ("Drawing {}".format (view))
            # convert into BGR so we can draw it
            fgs[view] = cv2.cvtColor (fgs[view], cv2.COLOR_GRAY2BGR)
            mask_combine[view] = np.zeros_like (fgs[view])
            for b in blobs : 
                corner = CBP.get_border_blobs (b)
                tan_bot = corner[0][1]
                tan_top = corner[1][1]

                corner = np.array (corner, np.int32)
                corner = corner.reshape ((-1, 1, 2))

                max_cnt = CBP.split_blob (b)
                # if max_cnt is None : 
                #     max_cnt = np.zeros_like (fgs[view])
                #     # max_cnt = cv2.cvtColor (max_cnt, cv2.COLOR_BGR2GRAY)
                # cv2.fillPoly (mask_combine[view], max_cnt, (120, 120, 120))
                mask_combine[view] += max_cnt

            # draw bounding box
            # fgs[view] = draw_bounding_box_contours (fgs[view], blobs) 

            # put view name in top of it
            loc = (10, 50)
            cv2.putText (fgs[view], '{} #{}'.format (view, tot_blobs[view]), loc, cv2.FONT_HERSHEY_PLAIN, 2, (0, 100, 255), 3) # and so we can insert text


        # fgs['and'] = cv2.cvtColor (fgs['and'], cv2.COLOR_GRAY2BGR)
        # fgs.pop ('and', None)
        fgs = np.hstack (fgs.values ())

        # dsts = np.hstack (dsts.values ())
        # dsts = CBP.color_quantization (dsts, K=6)

        # imgs = np.hstack (dsts.values ())

        # masks = np.hstack (CBP.masks.values ())

        mask_combine = np.hstack (mask_combine.values ())

        path_to_save = os.path.join (os.path.abspath (os.path.dirname (__file__)), 'imgs', str (data_id))
        if not os.path.isdir (path_to_save) : 
            os.mkdir (path_to_save)

        exp_name = 'poly-split-point'
        # cv2.imwrite (os.path.join (path_to_save, '{}.jpg'.format (exp_name)), mask_combine )
        # exp_name = 'tsi'
        # cv2.imwrite (os.path.join (path_to_save, '{}.jpg'.format (exp_name)), imgs)

        # csv_out.writerow (row)
        # continue
        while True : 
            cv2.imshow ('default', mask_combine)

            if (cv2.waitKey(1) & 0xFF == ord('q')) :
                sys.exit ()
            elif (cv2.waitKey (1) >= 0) : 
                break

    # write accuracy in the end
    row = {'data_id' : 'Total True'}
    for key in accuracy : 
        row[key] = accuracy[key]
    # csv_out.writerow (row)
    row = {'data_id' : 'Accuracy'}
    print ('Accuracy : ')
    for key in accuracy : 
        row[key] = accuracy[key] / len (CBP.GT_LIST_ID)
        print ('{} - {:.2F}'.format (key, row[key]))
    # csv_out.writerow (row)

if __name__ == '__main__' : 
    main ()
