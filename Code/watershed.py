'''
This file is to compute watershed given the seed image in the gvf.py. 

'''
import cv2
import numpy as np
from numpy import unique

class WATERSHED():
	'''
	This class contains all the function to compute watershed.

	'''
	def __init__(self, images, markers):
		self.images = images
		self.markers = markers

	def watershed_compute(self):
		'''
		This function is to compute watershed given the newimage and the seed image
		(center candidates). In this function, we use cv2.watershed to implement watershed.

		Input: newimage (height * weight * # of images)

		Output: watershed images (height * weight * # of images)

		'''
		result = []
		outmark = []
		outbinary = []

		for i in range(len(self.images)):
			# generate a 3-channel image in order to use cv2.watershed
			imgcolor = np.zeros((self.images[i].shape[0], self.images[i].shape[1], 3), np.uint8)
			for c in range(3): 
				imgcolor[:,:,c] = self.images[i]

			# compute marker image (labelling)
			if len(self.markers[i].shape) == 3:
				self.markers[i] = cv2.cvtColor(self.markers[i],cv2.COLOR_BGR2GRAY)
			_, mark = cv2.connectedComponents(self.markers[i])
			# mark = self.markers[i].astype(np.int32)
			# watershed!
			mark = cv2.watershed(imgcolor,mark)

			u, counts = unique(mark, return_counts=True)
			counter = dict(zip(u, counts))
			for i in counter:
				if counter[i] > 1200:
					mark[mark==i] = 0

			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
			mark = cv2.morphologyEx(mark.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
			_, mark = cv2.connectedComponents(mark.astype(np.uint8))

			# mark image and add to the result 
			temp = cv2.cvtColor(imgcolor,cv2.COLOR_BGR2GRAY)
			result.append(temp)
			outmark.append(mark.astype(np.uint8))

			binary = mark.copy()
			binary[mark>0] = 255
			outbinary.append(binary.astype(np.uint8))

		return result, outbinary, outmark

def main():
	'''
	This part is for testing watershed.py with single image.

	Input: an original image
	       a seeds image

	Output: Binary image after watershed

	'''
	images = []
	image = cv2.imread(sys.argv[1])
	images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
	markers = []
	marker = cv2.imread(sys.argv[2])
	markers.append(cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY))

	# watershed
    ws = WS(newimg, imgpair) 
    wsimage, binarymark, mark = ws.watershed_compute()

	cv2.imwrite('binarymark.tif', (np.clip(binarymark, 0, 255)).astype(np.uint8))

# if python says run, then we should run
if __name__ == '__main__':
    main()
	