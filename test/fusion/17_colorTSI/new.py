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
from matplotlib import pyplot as plt

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


    """
    ============================
    Object Detection Functionalitiy
    ============================
    """
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

    def process (self, imgs=None, iterations=1) :
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
            fg = process_morphological (fg, iterations)

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


    """
    ============================
    Split Contour Functionality
    ============================
    """
    def split_blob (self, poly, hull) : 
        mask_poly, mask_hull = self.create_convex_mask (poly, hull)
        max_cnt = self.get_largest_residu_contour (mask_poly, mask_hull)
        cutting_point = self.get_cutting_point (hull, max_cnt)
        contours = self.split_blob_from_cutting_point (mask_poly, cutting_point)

        return contours 

    def create_convex_mask (self, poly, hull) : 
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

        return mask_poly, mask_hull

    def get_largest_residu_contour (self, mask_poly, mask_hull) : 
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

        return max_cnt

    def get_residu_contour (self, mask_poly, mask_hull) : 
        # blob diff 
        blob_diff = cv2.bitwise_xor (mask_hull, mask_poly)
        im2, contours, hierarchy = cv2.findContours(
                blob_diff, 
                cv2.RETR_TREE, 
                cv2.CHAIN_APPROX_SIMPLE
            )

        return contours

    def get_cutting_point (self, hull, max_cnt) : 
        cutting_point = None
        most_inner_dist = None

        M = cv2.moments (hull)
        cx = int (M['m10'] / M['m00'])
        cy = int (M['m01'] / M['m00'])
        centroid = np.array ((cx, cy))

        for mc in max_cnt : 
            mc = np.array (mc[0])

            ext_dist = None 
            ext_point = None

            dist = np.linalg.norm (mc-centroid)

            if cutting_point is None or dist < most_inner_dist : 
                cutting_point = mc
                most_inner_dist = dist 

        return cutting_point

    def split_blob_from_cutting_point (self, mask_poly, cutting_point) : 
        # split horizontally
        up, bot = mask_poly[:cutting_point[1], :], mask_poly[cutting_point[1]+4:, :]
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
        up_area = max ([cv2.contourArea (c) for c in up] + [0]) # add [0] so its not empty
        bot_area = max ([cv2.contourArea (c) for c in bot] + [0])
        if up_area and bot_area : 
            horizontal_diff = abs (up_area - bot_area)
        else : 
            horizontal_diff = sys.maxsize

        # split vertically
        left, right = mask_poly[:, :cutting_point[0]], mask_poly[:, cutting_point[0]+4:]
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

        left_area = max ([cv2.contourArea (c) for c in left] + [0])
        right_area = max ([cv2.contourArea (c) for c in right] + [0])
        if left_area and right_area : 
            vertical_diff = abs (left_area - right_area)
        else : 
            vertical_diff = sys.maxsize 

        if horizontal_diff < vertical_diff : 
            # p1 = (0, cutting_point[1])
            # p2 = (mask_poly.shape[1], cutting_point[1])
            for b_idx, b in enumerate (bot) : 
                for p_idx, p in enumerate (b) : 
                    bot[b_idx][p_idx][0][1] = bot[b_idx][p_idx][0][1] + cutting_point[1]+4
                
            up.extend (bot)
            res = up
        else : 
            # p1 = (cutting_point[0], 0)
            # p2 = (cutting_point[0], mask_poly.shape[0])
            for l_idx, l in enumerate (right) : 
                for p_idx, p in enumerate (l) : 
                    right[l_idx][p_idx][0][0] = right[l_idx][p_idx][0][0] + cutting_point[0]+4
            right.extend (left)
            res = right

        clean_res = [ r for r in res if cv2.contourArea (r) >= 10 ]
        return  clean_res 

    def get_clean_blobs (self, blobs) :
        dirty_blobs = blobs
        clean_blobs = []
        while dirty_blobs : 
            need2split_blobs = []

            for b in dirty_blobs : 

                # get surrounding of poly and hull
                epsilon = 0.01*cv2.arcLength(b,True)
                poly = cv2.approxPolyDP(b,epsilon,True)
                hull = cv2.convexHull(b)

                # test weather it should be divided or not
                area_poly = cv2.contourArea (poly)
                area_hull = cv2.contourArea (hull)
                try :
                    ratio_area = area_poly / area_hull
                except ZeroDivisionError as e : 
                    continue

                if ratio_area >= 0.9 : 
                    # no need to divide
                    # cv2.fillPoly (mask_combine[view], [poly],  (125, 125, 125))
                    # cv2.drawContours (mask_combine[view], [hull], 0, (125, 125, 255), 2)
                    clean_blobs.append (b)
                    continue

                # cv2.drawContours (mask_combine[view], [hull], 0, (255, 0, 255), 2)
                sub_b = self.split_blob (poly, hull)
                if sub_b : 
                    for bb in sub_b : 
                        need2split_blobs.append (bb)
                else : 
                    clean_blobs.append (b)

            dirty_blobs = need2split_blobs
            # print (len (clean_blobs))

            # max_cnt = CBP.split_blob (b)
            # mask_combine[view] += max_cnt

        return clean_blobs

    def get_clean_blobs_once (self, blobs) : 
        clean_blobs = []
        for b in blobs : 
            # get surrounding of poly and hull
            epsilon = 0.01*cv2.arcLength(b,True)
            poly = cv2.approxPolyDP(b,epsilon,True)
            hull = cv2.convexHull(b)

            # test weather it should be divided or not
            area_poly = cv2.contourArea (poly)
            area_hull = cv2.contourArea (hull)
            try :
                ratio_area = area_poly / area_hull
            except ZeroDivisionError as e : 
                continue

            if ratio_area >= 0.8 : 
                # no need to divide
                # cv2.fillPoly (mask_combine[view], [poly],  (125, 125, 125))
                # cv2.drawContours (mask_combine[view], [hull], 0, (125, 125, 255), 2)
                clean_blobs.append (b)
                continue

            # cv2.drawContours (mask_combine[view], [hull], 0, (255, 0, 255), 2)
            sub_b = self.split_blob (poly, hull)
            if sub_b : 
                for bb in sub_b : 
                    clean_blobs.append (bb)
            else : 
                clean_blobs.append (b)

        return clean_blobs

    def split_vertical (self, blob, tot_lanes) : 
        (x,y,w,h) = cv2.boundingRect (blob)
        if tot_lanes == 2 : 
            split_idx = 150 
        else : 
            if y <= 100 <= h :
                split_idx = 100
            else : 
                split_idx = 200

        # generate blob
        shape = list (self.shape)
        shape.append (3)
        tmp = np.zeros (shape[:3]).astype ('uint8')
        cv2.fillPoly (tmp, [blob], (255, 255, 255))
        # grayscale
        tmp = cv2.cvtColor (tmp, cv2.COLOR_BGR2GRAY)
        # split figuratively
        tmp[split_idx-3:split_idx+3, :] = 0
        # get contour
        blobs = TSIUtil.get_contours (tmp)
        return blobs

    def split_horizontal (self, blobs) : 
        # first find the largest area of
        return blobs


    @staticmethod
    def is_horizontal_splittable (blob, largest_contour) : 
        (x,y,w,h) = cv2.boundingRect (blob)
        (xx, yy, ww, hh) = cv2.boundingRect (largest_contour)
        M = cv2.moments(largest_contour)
        try : 
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        except ZeroDivisionError as e : 
            return False
        if x+0.3*w <= cx <= x+0.7*w and yy+hh >= y+(h/2) : 
            return True
        else  : 
            return False

    @staticmethod
    def find_tip_point (blob) : 
        min_y = None
        min_score = None
        for c in blob : 
            c = c[0]

            if min_score is None or c[1] < min_score : 
                min_score = min_y
                min_y = c

        return min_y


    """
    ============================
    Feature Extraction Functionality
    ============================
    """
    @staticmethod
    def extract_feature_from_masks (img, contour, bins=20) : 
        # first extract region blob from image
        img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros (img.shape[:2], np.uint8)
        cv2.fillPoly (mask, [contour], (255, 255, 255) )

        # hist = cv2.calcHist ([img], [0,1,2], mask, [bins] * 3, [0, 256] * 3)
        hist = cv2.calcHist ([img], [0], mask, [bins], [0, 256])
        hist = hist.flatten () # flatten
        return hist

    @staticmethod
    def cluster_histogram (histogram, centroids, radius=70, thres=0.8) : 
        data2cluster = defaultdict (lambda : None)
        cluster2data = defaultdict (lambda : set ())

        for h_idx, h in enumerate (histogram) : 
            if data2cluster[h_idx] is None : 
                data2cluster[h_idx] = h_idx
                cluster2data[h_idx] |= set ([h_idx])

            for i_idx in range (h_idx+1, len (histogram)) : 
                i = histogram[i_idx]
                # check distance
                centroid_distance = np.linalg.norm (centroids[h_idx] - centroids[i_idx])

                if centroid_distance >= radius : 
                    continue

                # check for histogram similarity
                score = cv2.compareHist(h, i, cv2.HISTCMP_CORREL)

                if score >= thres : 
                    # merging
                    if data2cluster[i_idx] is None : 
                        data2cluster[i_idx] = data2cluster[h_idx]
                        cluster2data[h_idx] |= set ([i_idx])

                    else : 
                        for member in cluster2data[i_idx] :  
                            data2cluster[member] = data2cluster[h_idx]
                        cluster2data.pop (i_idx, None)

        return cluster2data

    """
    ============================
    Misc Utility
    ============================
    """
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
        if len (fg.shape) == 3 and len (mask.shape) != 3 : 
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

    def process (self, imgs=None, iterations=1) :
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
            fg = process_morphological (fg, iterations=1)

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

def main_1 () : 
    # CBP = MultiTSIPFM (w_color=True)
    CBP = ColorBasedPFM (w_color=False)

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
        clean_blobs_all = {}
        mask_combine = {}
        mask_combine['and'] = None

        for view in VIEW :

            blobs = TSIUtil.get_contours (fgs[view], min_width=50, min_area=100)
            tot_blobs[view] = len (blobs)

            row[view] = tot_blobs[view]

            # calculate accuracy
            if int (gt['GT']) == len (blobs) : 
                accuracy[view] += 1
            else : 
                accuracy[view] += 0


            logger.info ("Drawing {}".format (view))
            # convert into BGR so we can draw it
            fgs[view] = cv2.cvtColor (fgs[view], cv2.COLOR_GRAY2BGR)
            mask_combine[view] = np.zeros_like (fgs[view])

            clean_blobs = CBP.get_clean_blobs (blobs)
            clean_blobs_all[view] = clean_blobs

            random_color = get_colors (len (clean_blobs))
            for b_idx, b in enumerate (clean_blobs) : 
                b_hull = cv2.convexHull (b)
                # cv2.drawContours (mask_combine[view], [b_hull], 0, random_color[b_idx], 2)
                cv2.fillPoly (mask_combine[view], [b_hull], (255,255,255))
                # cv2.fillPoly (mask_combine['and'], [b_hull], 0, (85,85,85), 2)

            if mask_combine['and'] is None : 
                mask_combine['and'] = mask_combine[view]
            else : 
                mask_combine['and'] = cv2.bitwise_and (mask_combine['and'], mask_combine[view])

            # mask_combine[view] = CBP.get_blobs_from_image (dsts[view], mask_combine[view])

            # for b_idx, b in enumerate (clean_blobs) : 
            #     b_hull = cv2.convexHull (b)
            #     cv2.drawContours (mask_combine[view], [b_hull], 0, random_color[b_idx], 2)
            #     # cv2.fillPoly (mask_combine[view], [b_hull], (255,255,255))

            # extract blobs
            # logger.info ("Get contours for {}".format (view))
            # dsts[view] = CBP.get_blobs_from_image (dsts[view], fgs[view])
            # for i in range (3) : 
            #     dsts[view] = cv2.medianBlur (dsts[view], 5)
            # dsts[view] = CBP.color_quantization (dsts[view], K=5)
            # dsts[view] = CBP.apply_sobel (dsts[view])

            # draw bounding box
            # fgs[view] = draw_bounding_box_contours (fgs[view], blobs) 

            # put view name in top of it
            loc = (10, 50)
            cv2.putText (fgs[view], '{} #{}'.format (view, tot_blobs[view]), loc, cv2.FONT_HERSHEY_PLAIN, 2, (0, 100, 255), 3) # and so we can insert text


        # fgs['and'] = cv2.cvtColor (fgs['and'], cv2.COLOR_GRAY2BGR)
        # fgs.pop ('and', None)
        # fgs = np.hstack (fgs.values ())

        if len (mask_combine['and'].shape) == 3 : 
            mask_and = cv2.cvtColor (mask_combine['and'], cv2.COLOR_BGR2GRAY)
        else : 
            mask_and = mask_combine['and']
        mask_and = cv2.threshold (mask_and, 25, 255, cv2.THRESH_BINARY)[1]

        im2, blob_combine, hierarchy = cv2.findContours(
                mask_and, 
                cv2.RETR_TREE, 
                cv2.CHAIN_APPROX_SIMPLE
            )

        features = []
        centroids = []
        thres = 70
        for b_idx, blob in enumerate (blob_combine) : 
            fhist_combine = None
            for view in (['right']) : 
                fhist = CBP.extract_feature_from_masks (dsts[view], blob)
                if fhist_combine is None : 
                    fhist_combine = fhist
                else : 
                    fhist_combine = np.hstack ((fhist_combine, fhist))

            features.append (fhist_combine)

            M = cv2.moments (blob)
            try : 
                cx = int (M['m10'] / M['m00'])
                cy = int (M['m01'] / M['m00'])
            except ZeroDivisionError : 
                cx = -1
                cy = -1
            centroid = np.array ((cx, cy))
            centroids.append (centroid)


        cluster = CBP.cluster_histogram (features, centroids)
        colors = get_colors (len (cluster.keys ()))
        mask_combine['cluster'] = np.zeros_like (dsts[view])
        for cl_idx,  cl in enumerate (cluster.keys ()) : 
            this_color = colors[cl_idx]
            for blob_idx in cluster[cl] : 
                cv2.fillPoly (mask_combine['cluster'], [blob_combine[blob_idx]], this_color)

        # mask_combine['rest'] = CBP.get_blobs_from_image (dsts['center'], mask_combine['and'])
        # mask_combine['real'] = dsts['center']
        # mask_combine = np.hstack ((mask_combine[k] for k in ('real', 'rest', 'and', 'left', 'center', 'right')))
        # mask_combine = np.vstack ((mask_combine, dsts))

        path_to_save = os.path.join (os.path.abspath (os.path.dirname (__file__)), 'imgs', str (data_id))
        if not os.path.isdir (path_to_save) : 
            os.mkdir (path_to_save)

        exp_name = 'tsi_clustering'
        cv2.imwrite (os.path.join (path_to_save, '{}.jpg'.format (exp_name)), mask_combine['cluster'])
        # exp_name = 'tsi_default'
        # cv2.imwrite (os.path.join (path_to_save, '{}.jpg'.format (exp_name)), dsts)

        csv_out.writerow (row)
        continue
        while True : 
            cv2.imshow ('default', mask_combine['cluster'])

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

def draw_blobs (img, blobs) : 
    random_color = get_colors (2)
    for b_idx, b in enumerate (blobs) : 
        hull = cv2.convexHull (b)
        epsilon = 0.01*cv2.arcLength(b,True)
        poly = cv2.approxPolyDP(b,epsilon,True)

        cv2.fillPoly (img, [hull], (125, 125, 125))
        cv2.fillPoly (img, [poly], (255, 255, 255))
        cv2.drawContours (img, [poly], 0, (125, 125, 125), 2)

    return img

def main_2 () : 
    TOT_LANES = [3, 2, 2, None, 2, 2, 3]
    CBP = ColorBasedPFM (w_color=True)
    # CBP = MultiTSIPFM (w_color=True)

    csv_out = csv.DictWriter (open ('result.csv', 'wb'), ('data_id', 'left', 'center', 'right', 'and'))
    csv_out.writeheader ()

    accuracy = defaultdict (lambda : 0) 
    for data_id in CBP.GT_LIST_ID[:] : 

        row = {}
        row['data_id'] = data_id

        CBP.set_scene (data_id)
        gt = [ ent for ent in CBP.GT_ONE_FRAME if int (ent['data_id'].split ('-')[0]) == data_id ][0]
        ses_id = int (gt['data_id'].split ('-')[1])

        # skip to the end only
        imgs = {}
        while True : 
            try :
                imgs, dsts, fgs = CBP.process ()
            except StopIteration as e : 
                break

        mask_combine = {}
        frame = None
        for view in fgs.keys () :
            # """

            # divide by lane
            # if TOT_LANES[ses_id] == 3 : 
            #     fgs[view][95:105, :] = 0
            #     fgs[view][185:205, :] = 0
            # else : 
            #     fgs[view][145:155, :] = 0

            blobs = TSIUtil.get_contours (fgs[view], min_width=50, min_area=100)
            fgs[view] = cv2.cvtColor (fgs[view], cv2.COLOR_GRAY2BGR)
            # blobs = CBP.get_clean_blobs_once (blobs)
            fgs[view] = draw_blobs (fgs[view], blobs)
            loc = (10, 50)
            cv2.putText (fgs[view], '{} #{}'.format (view, len (blobs)), loc, cv2.FONT_HERSHEY_PLAIN, 2, (0, 100, 255), 2) # and so we can insert text
            # mask_combine[view] = draw_blobs (mask_combine[view], blobs)
            row[view] = len (blobs)

            # calculate accuracy
            if int (gt['GT']) == len (blobs) : 
                accuracy[view] += 1
            else : 
                accuracy[view] += 0

        keys = fgs.keys ()
        keys.insert (0, 'real')
        fgs['real'] = dsts['center']
        loc = (10, 50)
        cv2.putText (fgs['real'], '{} #{}'.format ('real', gt['GT']), loc, cv2.FONT_HERSHEY_PLAIN, 2, (0, 100, 255), 2) # and so we can insert text


        # csv_out.writerow (row)
        # continue
        path_to_save = os.path.join (os.path.abspath (os.path.dirname (__file__)), 'imgs', str (data_id))
        if not os.path.isdir (path_to_save) : 
            os.mkdir (path_to_save)

        exp_name = 'analyze_contour_PFM_1'
        cv2.imwrite (os.path.join (path_to_save, '{}.jpg'.format (exp_name)), np.vstack ([fgs[k] for k in keys]))
        # cv2.imwrite (os.path.join (path_to_save, '{}-and.jpg'.format (exp_name)), fgs['and'])
        continue
        while True : 
            cv2.imshow ('default', frame)

            if (cv2.waitKey(1) & 0xFF == ord('q')) :
                sys.exit ()
            elif (cv2.waitKey (1) >= 0) : 
                break

    # write accuracy in the end
    row = {'data_id' : 'Total True'}
    for key in accuracy : 
        row[key] = accuracy[key]
    csv_out.writerow (row)
    row = {'data_id' : 'Accuracy'}
    print ('Accuracy : ')
    for key in accuracy : 
        row[key] = accuracy[key] / len (CBP.GT_LIST_ID)
        print ('{} - {:.2F}'.format (key, row[key]))
    csv_out.writerow (row)

def main_3 () : 
    TOT_LANES = [3, 2, 2, None, 2, 2, 3]
    # CBP = ColorBasedPFM (w_color=True)
    CBP = MultiTSIPFM (w_color=True)

    for data_id in CBP.GT_LIST_ID[:1] : 
        CBP.set_scene (data_id)
        gt = [ ent for ent in CBP.GT_ONE_FRAME if int (ent['data_id'].split ('-')[0]) == data_id ][0]
        ses_id = int (gt['data_id'].split ('-')[1])
        ses_length_line = 300 / TOT_LANES[ses_id]

        # skip to the end only
        imgs = {}
        while True : 
            try :
                imgs, dsts, fgs = CBP.process ()
            except StopIteration as e : 
                break

        mask_residu = {} 
        largest_contour = []
        for view in ('and', 'left', 'right', 'center') :
            mask_residu[view] = []
            blobs = TSIUtil.get_contours (fgs[view], min_width=50, min_area=100)
            blobs_splitted_vertical = []

            if view == 'and' : 
                for b in blobs : 
                    # check for height
                    (x,y, w, h) = cv2.boundingRect (b)
                    if h >= ses_length_line + (1/2 * ses_length_line) : 
                        # draw here
                        blobs_splitted_vertical.extend (CBP.split_vertical (b, tot_lanes=TOT_LANES[ses_id]))

                    else : 
                        blobs_splitted_vertical.append (b)

                blobs = blobs_splitted_vertical

                for b in blobs : 
                    # get hull an poligon approximation
                    b_hull = cv2.convexHull (b)
                    epsilon = 0.01*cv2.arcLength(b,True)
                    b_poly = cv2.approxPolyDP(b,epsilon,True)

                    # get mask
                    mask_poly, mask_hull = CBP.create_convex_mask (b_poly, b_hull)
                    
                    # get largets contour of residu
                    this_largest_residu = CBP.get_largest_residu_contour (mask_poly, mask_hull)
                    if CBP.is_horizontal_splittable (b, this_largest_residu) : 
                        largest_contour.append (this_largest_residu)

                fgs['aftersplit'] = np.zeros_like (dsts['center'])
                fgs['aftersplit'] = draw_blobs (fgs['aftersplit'], blobs)


            # get something actually split
            for lg in largest_contour : 
                mask_residu[view].append (np.zeros (fgs[view].shape[:2]).astype ('uint8'))
                cv2.drawContours(mask_residu[view][-1],[lg], 0, (255,255,255), -1)
                mask_residu[view][-1] = cv2.threshold (mask_residu[view][-1], 25, 255, cv2.THRESH_BINARY)[1]

                inverted_fgs = cv2.bitwise_not (cv2.threshold (fgs[view], 25, 255, cv2.THRESH_BINARY)[1])
                mask_residu[view][-1] = cv2.bitwise_and (inverted_fgs, mask_residu[view][-1])

            fgs[view] = cv2.cvtColor (fgs[view], cv2.COLOR_GRAY2BGR)
            fgs[view] = draw_blobs (fgs[view], blobs)

            loc = (10, 50)
            cv2.putText (fgs[view], '{}'.format (view[0], len (blobs)), loc, cv2.FONT_HERSHEY_PLAIN, 2, (0, 100, 255), 2) # and so we can insert text

        path_to_save = os.path.join (os.path.abspath (os.path.dirname (__file__)), 'imgs', str (data_id))
        exp_name = 'split_vertical'

        for lg_idx, lg in enumerate (largest_contour) :  
            m_biner = np.zeros (mask_residu[view][lg_idx].shape)
            for view in VIEW : 
                m_biner += mask_residu[view][lg_idx]

            tot_view = len (m_biner[ np.where (m_biner > 300) ])
            tot_and = len (mask_residu['and'][lg_idx] [np.where (m_biner > 1) ])
            print ('id : {}, lg_idx : {}, {:.2F}'.format (data_id, lg_idx, tot_view / tot_and))

            if tot_view / tot_and >= 0.70 : 
                for view in list (fgs.keys ()) : 
                    cv2.drawContours (fgs[view], [lg], 0, (255, 0, 255), -1)

        # frame = np.hstack ([mask_residu[k] for k in mask_residu.keys ()])
        # frame = cv2.cvtColor (frame, cv2.COLOR_GRAY2BGR)
        # frame = np.hstack ((frame, dsts['center']))
        fgs.pop ('and', None)
        fgs.pop ('aftersplit', None)
        cv2.imwrite (os.path.join ('dd.jpg'.format (exp_name)), np.hstack (fgs.values ()))

def test_PFM (iterations=1) : 
    TOT_LANES = [3, 2, 2, None, 2, 2, 3]
    CBP = ColorBasedPFM (w_color=True)

    path_to_save = os.path.join (os.path.abspath (os.path.dirname (__file__)), 'PFM_{}'.format (iterations))
    if not os.path.isdir (path_to_save) : 
        os.mkdir (path_to_save)

    csv_out = csv.DictWriter (open (os.path.join (path_to_save, 'result.csv'), 'wb'), ('data_id', 'left', 'center', 'right', 'and'))
    csv_out.writeheader ()

    accuracy = defaultdict (lambda : 0) 
    for data_id in CBP.GT_LIST_ID : 
        row = {}
        row['data_id'] = data_id

        CBP.set_scene (data_id)
        gt = [ ent for ent in CBP.GT_ONE_FRAME if int (ent['data_id'].split ('-')[0]) == data_id ][0]
        ses_id = int (gt['data_id'].split ('-')[1])
        ses_length_line = 300 / TOT_LANES[ses_id]

        # skip to the end only
        imgs = {}
        while True : 
            try :
                imgs, dsts, fgs = CBP.process (iterations=iterations)
            except StopIteration as e : 
                break


        order = ('and', 'left', 'right', 'center')
        blobs_all = {}
        for view in  order :
            blobs = TSIUtil.get_contours (fgs[view], min_width=50, min_area=100)
            blobs_all[view] = blobs

            # coloring
            fgs[view] = cv2.cvtColor (fgs[view], cv2.COLOR_GRAY2BGR)
            loc = (10, 30)
            cv2.putText (fgs[view], '{} #{}'.format (view[0], len (blobs)), loc, cv2.FONT_HERSHEY_PLAIN, 2, (0, 100, 255), 3) # and so we can insert text
            fgs[view] = draw_bounding_box_contours (fgs[view], blobs) 

            # accuracy
            row[view] = len (blobs)

            # calculate accuracy
            if int (gt['GT']) == len (blobs) : 
                accuracy[view] += 1
            else : 
                accuracy[view] += 0

        cv2.imwrite ('{}.jpg'.format (os.path.join (path_to_save, str (data_id))), np.vstack ([fgs[v] for v in order]))
        dsts['center'] = draw_bounding_box_contours (dsts['center'], blobs_all['and'])
        cv2.imwrite ('{}-center.jpg'.format (os.path.join (path_to_save, str (data_id))), dsts['center'])
            
        csv_out.writerow (row)

    row = {'data_id' : 'Total True'}
    for key in accuracy : 
        row[key] = accuracy[key]
    csv_out.writerow (row)
    row = {'data_id' : 'Accuracy'}
    print ('Accuracy : ')
    for key in accuracy : 
        row[key] = accuracy[key] / len (CBP.GT_LIST_ID)
        print ('{}. {}/{} - {:.2F}'.format (key, accuracy[key], len (CBP.GT_LIST_ID) - accuracy[key], row[key]))
    csv_out.writerow (row)

def test_TSI (iterations=1) : 
    TOT_LANES = [3, 2, 2, None, 2, 2, 3]
    CBP = MultiTSIPFM (w_color=True)

    path_to_save = os.path.join (os.path.abspath (os.path.dirname (__file__)), 'TSI_{}'.format (iterations))
    if not os.path.isdir (path_to_save) : 
        os.mkdir (path_to_save)

    csv_out = csv.DictWriter (open (os.path.join (path_to_save, 'result.csv'), 'wb'), ('data_id', 'left', 'center', 'right', 'and'))
    csv_out.writeheader ()

    accuracy = defaultdict (lambda : 0) 
    for data_id in CBP.GT_LIST_ID : 

        row = {}
        row['data_id'] = data_id

        CBP.set_scene (data_id)
        gt = [ ent for ent in CBP.GT_ONE_FRAME if int (ent['data_id'].split ('-')[0]) == data_id ][0]
        ses_id = int (gt['data_id'].split ('-')[1])
        ses_length_line = 300 / TOT_LANES[ses_id]

        # skip to the end only
        imgs = {}
        while True : 
            try :
                imgs, dsts, fgs = CBP.process (iterations=iterations)
            except StopIteration as e : 
                break

        path_to_save = os.path.join (os.path.abspath (os.path.dirname (__file__)), 'TSI_{}'.format (iterations))
        if not os.path.isdir (path_to_save) : 
            os.mkdir (path_to_save)

        order = ('and', 'left', 'right', 'center')
        for view in  order :
            blobs = TSIUtil.get_contours (fgs[view], min_width=50, min_area=100)
            fgs[view] = cv2.cvtColor (fgs[view], cv2.COLOR_GRAY2BGR)
            loc = (10, 30)
            cv2.putText (fgs[view], '{} #{}'.format (view[0], len (blobs)), loc, cv2.FONT_HERSHEY_PLAIN, 2, (0, 100, 255), 3) # and so we can insert text
            fgs[view] = draw_bounding_box_contours (fgs[view], blobs) 

            row[view] = len (blobs)

            # calculate accuracy
            if int (gt['GT']) == len (blobs) : 
                accuracy[view] += 1
            else : 
                accuracy[view] += 0

        cv2.imwrite ('{}.jpg'.format (os.path.join (path_to_save, str (data_id))), np.hstack ([fgs[v] for v in order]))

    row = {'data_id' : 'Total True'}
    for key in accuracy : 
        row[key] = accuracy[key]
    csv_out.writerow (row)
    row = {'data_id' : 'Accuracy'}
    print ('Accuracy : ')
    for key in accuracy : 
        row[key] = accuracy[key] / len (CBP.GT_LIST_ID)
        print ('{}. {}/{} - {:.2F}'.format (key, accuracy[key], len (CBP.GT_LIST_ID) - accuracy[key], row[key]))
    csv_out.writerow (row)

def test_TSI_split (iterations=1):

    def split_horizontal (blobs) : 
        blobs_splitted_vertical = []
        for b in blobs : 
            # check for height
            (x,y, w, h) = cv2.boundingRect (b)
            if h >= ses_length_line + (1/2 * ses_length_line) : 
                # draw here
                blobs_splitted_vertical.extend (CBP.split_vertical (b, tot_lanes=TOT_LANES[ses_id]))

            else : 
                blobs_splitted_vertical.append (b)

        return blobs_splitted_vertical
    
    TOT_LANES = [3, 2, 2, None, 2, 2, 3]
    CBP = MultiTSIPFM (w_color=True)

    path_to_save = os.path.join (os.path.abspath (os.path.dirname (__file__)), 'TSI_SPLIT_{}'.format (iterations))
    if not os.path.isdir (path_to_save) : 
        os.mkdir (path_to_save)

    csv_out = csv.DictWriter (open (os.path.join (path_to_save, 'result.csv'), 'wb'), ('data_id', 'left', 'center', 'right', 'and'))
    csv_out.writeheader ()

    accuracy = defaultdict (lambda : 0) 
    for data_id in CBP.GT_LIST_ID : 

        row = {}
        row['data_id'] = data_id

        CBP.set_scene (data_id)
        gt = [ ent for ent in CBP.GT_ONE_FRAME if int (ent['data_id'].split ('-')[0]) == data_id ][0]
        ses_id = int (gt['data_id'].split ('-')[1])
        ses_length_line = 300 / TOT_LANES[ses_id]

        # skip to the end only
        imgs = {}
        while True : 
            try :
                imgs, dsts, fgs = CBP.process (iterations=iterations)
            except StopIteration as e : 
                break

        mask_residu = defaultdict (list) 
        largest_contour = []
        largest_contour_idx = []
        order = ('and', 'left', 'right', 'center')
        blobs_splitted = []
        blobs_and = []
        for view in  order :
            blobs = TSIUtil.get_contours (fgs[view], min_width=50, min_area=100)

            if view == 'and' : 

                blobs = split_horizontal (blobs)
                blobs_and = blobs[:]
                for b_idx, b in enumerate (blobs) : 
                    # get hull an poligon approximation
                    b_hull = cv2.convexHull (b)
                    epsilon = 0.01*cv2.arcLength(b,True)
                    b_poly = cv2.approxPolyDP(b,epsilon,True)

                    # get mask
                    mask_poly, mask_hull = CBP.create_convex_mask (b_poly, b_hull)
                    
                    # get largets contour of residu
                    this_largest_residu = CBP.get_largest_residu_contour (mask_poly, mask_hull)
                    if CBP.is_horizontal_splittable (b, this_largest_residu) : 
                        largest_contour.append (this_largest_residu)
                        largest_contour_idx.append (b_idx)

                # fgs['aftersplit'] = np.zeros_like (dsts['center'])
                # fgs['aftersplit'] = draw_blobs (fgs['aftersplit'], blobs)
                blobs_splitted = blobs

            # get something actually split
            for lg in largest_contour : 

                mask_residu[view].append (np.zeros (fgs[view].shape[:2]).astype ('uint8'))
                cv2.drawContours(mask_residu[view][-1],[lg], 0, (255,255,255), -1)
                mask_residu[view][-1] = cv2.threshold (mask_residu[view][-1], 25, 255, cv2.THRESH_BINARY)[1]

                inverted_fgs = cv2.bitwise_not (cv2.threshold (fgs[view], 25, 255, cv2.THRESH_BINARY)[1])
                mask_residu[view][-1] = cv2.bitwise_and (inverted_fgs, mask_residu[view][-1])

            fgs[view] = cv2.cvtColor (fgs[view], cv2.COLOR_GRAY2BGR)
            # fgs[view] = draw_blobs (fgs[view], blobs)

            # loc = (10, 50)
            # cv2.putText (fgs[view], '{}'.format (view[0], len (blobs)), loc, cv2.FONT_HERSHEY_PLAIN, 2, (0, 100, 255), 2) # and so we can insert text

        skipped_blob = []
        for lg_idx, lg in enumerate (largest_contour) :  
            m_biner = np.zeros (mask_residu[view][lg_idx].shape)
            for view in VIEW : 
                m_biner += mask_residu[view][lg_idx]

            tot_view = len (m_biner[ np.where (m_biner > 300) ])
            tot_and = len (mask_residu['and'][lg_idx] [np.where (m_biner > 1) ])
            # print ('id : {}, lg_idx : {}, {:.2F}'.format (data_id, lg_idx, tot_view / tot_and))

            if tot_view / tot_and >= 0.50 and ses_id not in (29, 32, 35) : 
                # for view in list (fgs.keys ()) : 
                    # cv2.drawContours (fgs[view], [lg], 0, (255, 0, 255), -1)

                mask_blob = np.zeros (fgs[view].shape[:2]).astype ('uint8')
                mask_blob = cv2.drawContours (
                        mask_blob, 
                        [blobs_splitted[largest_contour_idx[lg_idx]]], 
                        0, 
                        (255, 255, 255), 
                        -1
                    )

                # get center 
                M = cv2.moments(lg)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                mask_blob[:, cx-5:cx+5] = 0
                blobs_splitted.extend (TSIUtil.get_contours (mask_blob))
                # its no longer valid, since we split it already
                skipped_blob.append (largest_contour_idx[lg_idx])

        blobs_splitted = [b for (idx, b) in enumerate (blobs_splitted) if idx not in skipped_blob]


        # row[view] = len (blobs)

        # calculate accuracy
        if int (gt['GT']) == len (blobs) : 
            accuracy[view] += 1
        else : 
            accuracy[view] += 0

        # draw
        print (len (blobs_splitted))
        fgs['aftersplit'] = np.zeros_like (dsts['center'])
        for b in blobs_splitted : 
            fgs['aftersplit'] = cv2.drawContours (fgs['aftersplit'], [b], 0, (255, 255, 255), -1)
        fgs['aftersplit'] = draw_bounding_box_contours (fgs['aftersplit'], blobs_splitted)
        dsts['center'] = draw_bounding_box_contours (dsts['center'], blobs_splitted)

        cv2.imwrite (
                '{}-center.jpg'.format (os.path.join (path_to_save, str (data_id))), 
                dsts['center']
            )


    row = {'data_id' : 'Total True'}
    for key in accuracy : 
        row[key] = accuracy[key]
    csv_out.writerow (row)
    row = {'data_id' : 'Accuracy'}
    print ('Accuracy : ')
    for key in accuracy : 
        row[key] = accuracy[key] / len (CBP.GT_LIST_ID)
        print ('{}. {}/{} - {:.2F}'.format (key, accuracy[key], len (CBP.GT_LIST_ID) - accuracy[key], row[key]))
    csv_out.writerow (row)

if __name__ == '__main__' : 
    # main_3 ()
    # test_PFM (2)
    # test_TSI (1)
    test_TSI_split (2)
