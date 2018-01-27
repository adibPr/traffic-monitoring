#!/usr/bin/env python2

import cv2

class FrameIterator (object) : 

    
    def __init__ (self, fpath) : 
        self.input_video = cv2.VideoCapture (fpath)

    def __iter__ (self) : 
        return self

    def __next__ (self) : 
        if self.input_video.isOpened () : 
            ret, frame = self.input_video.read ()
            return frame
        else : 
            raise StopIteration

    def next (self) : 
        return self.__next__ ()

if __name__ == '__main__' :
    fpath = './data/sample/session0_center.avi'
    fi = FrameIterator (fpath)
    while (True) : 
        img = next (fi)

        cv2.imshow ('default', img)

        if cv2.waitKey(1) & 0xFF == ord('q') :
            break
