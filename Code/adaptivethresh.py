'''
This file is to compute adaptive thresholding of image sequence in 
order to generate binary image for Nuclei segmentation.

Problem:
Due to the low contrast of original image, the adaptive thresholding is not working. 
Therefore, we change to regular threshold with threshold value as 129.
'''
import cv2
import sys
import numpy as np

class ADPTIVETHRESH():
    '''
    This class is to provide all function for adaptive thresholding.

    '''
    def __init__(self, images):
        self.images = []
        for img in images:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.images.append(img.copy())

    def applythresh(self, threshold = 129):
        '''
        applythresh function is to convert original image to binary image by thresholding.

        Input: image sequence. E.g. [image0, image1, ...]

        Output: image sequence after thresholding. E.g. [image0, image1, ...]
        '''
        out = []
        markers = []
        binarymark = []

        for img in self.images:
            img = cv2.GaussianBlur(img,(5,5),0).astype(np.uint8)
            _, thresh = cv2.threshold(img,threshold,1,cv2.THRESH_BINARY)

            # Using morphlogical operations to imporve the quality of result
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            out.append(thresh)

        return out

def main():
    '''
    This part is for testing adaptivethresh.py with single image.

    Input: an original image

    Output: Thresholding image

    '''
    img = [cv2.imread(sys.argv[1])]
    adaptive = ADPTIVETHRESH(img)
    thresh, markers = adaptive.applythresh(10)
    cv2.imwrite("adaptive.tiff", thresh[0]*255)

# if python says run, then we should run
if __name__ == '__main__':
    main()