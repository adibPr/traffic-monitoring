#!/usr/bin/env python2

import cv2
import os

class FrameIterator (object) :


    def __init__ (self, fpath) :
        self.fpath = fpath
        if os.path.isdir (fpath) :
            # then its a frame
            # create iterator of file
            self.gen = self.listdir ()

            # define class iterator
            def get_next () :
                for next_img in self.gen :
                    if next_img is not None :
                        return next_img
                    else :
                        raise StopIteration

        else :
            # then its a video
            self.input_video = cv2.VideoCapture (fpath)
            self.fps = self.input_video.get (cv2.CAP_PROP_FPS)

            def get_next () :
                if self.input_video.isOpened () :
                    ret, frame = self.input_video.read ()
                    return frame
                else :
                    raise StopIteration

        self.get_next = get_next

    def __iter__ (self) :
        return self

    def __next__ (self) :
        return self.get_next ()


    def next (self) :
        return self.__next__ ()

    def listdir (self) :
        # sort first based on its path
        paths = sorted (os.listdir (self.fpath), key=lambda x: int (x.split ('.')[0]))

        # iterate each image
        for img in paths :
            frame_path_full = os.path.join (self.fpath, img )
            this_img = cv2.imread (frame_path_full, 1)

            yield this_img

        yield None

    def skip (self, n=1) : 
        for i in range (n) : 
            print ('Skipping - {}'.format (i))
            self.__next__ ()

if __name__ == '__main__' :
    # fpath = './data/sample/session0_center.avi' # for video
    fpath = './data/sample/session0_center' # for image
    fi = FrameIterator (fpath)
    while (True) :
        img = next (fi)

        cv2.imshow ('default', img)

        if cv2.waitKey(1) & 0xFF == ord('q') :
            break
