#!/usr/bin/env python2

from __future__ import print_function, division

import cv2

def process_morphological (fg_binary) :

    # morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))

    # noise removal
    result = cv2.morphologyEx(fg_binary, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return result

def get_contours (frame_binary, min_area=400) :
    contours_selected = []

    im2, contours, hierarchy = cv2.findContours(
            frame_binary, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )

    # select contour that greater than the specified min_area and
    # not part of other contour
    for c_idx, c in enumerate (contours) :
        if cv2.contourArea(c) > min_area and hierarchy[0][c_idx][3] < 0 :
            contours_selected.append (c)

    return contours_selected

def draw_bounding_box_contours (frame, contours) :
    frame_cp = frame.copy ()
    for c_idx, c in enumerate (contours) :
        (x,y, w, h) = cv2.boundingRect (c)
        cv2.rectangle (frame_cp, (x,y), (x+w, y+h), (0, 255, 0), 2)
    return frame_cp


if __name__ == '__main__' : 
    from iterator import FrameIterator
    from background import BackgroundModel

    fi = FrameIterator ('./data/sample/session0_center.avi')
    bg = BackgroundModel (fi)

    print ("Learning")
    bg.learn (tot_frame_init=100)
    print ("Done")

    while True : 
        img = next (fi)
        fg = bg.apply (img)

        # remove shadows, i.e value 127
        fg = cv2.threshold (fg, 200, 255, cv2.THRESH_BINARY)[1]
    
        # remove noise
        fg = process_morphological (fg)

        # get blobs
        blobs = get_contours (fg)

        # draw bounding box
        frame = draw_bounding_box_contours (img, blobs)

        cv2.imshow ('default', frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')) :
            break

    cv2.destroyAllWindows ()
