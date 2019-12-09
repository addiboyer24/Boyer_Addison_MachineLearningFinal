import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np

class Webcam(object):
    # Webcam parameters here
    def __init__(self):
        pass
    def capture_video(self):
        video_capture = cv2.VideoCapture(0)
        '''
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 28)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 28)
        '''
        
        while True:
            #Capture the Video
            ret, frame = video_capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('Video', frame)
            

            if(cv2.waitKey(1) & 0xFF == ord('q')):
               break

            cv2.imwrite("./image.png", frame)

                                                  
        video_capture.release()
        cv2.destroyAllWindows()

    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def main(argv):
    #Declare Webcamera object
    
    camera = Webcam()
    camera.capture_video()
    im = rgb2gray(plt.imread("./image.png"))
    print(im.shape)
    plt.imshow(im, cmap='gray')
    plt.show()
    


if(__name__ == '__main__'):
    main(sys.argv[1:])
