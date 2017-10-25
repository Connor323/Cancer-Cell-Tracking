'''
This file is to compute gradient vector field (GVF) and then find the Nuclei center 
with the GVF result. 

'''
import cv2
import numpy as np
import sys
from adaptivethresh import ADPTIVETHRESH as athresh
from scipy import spatial as sp
from scipy import ndimage
from scipy.spatial import distance

looplimit = 500

def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True

class GVF():
	'''
	This class contains all function for calculating GVF and its following steps.

	'''
	def __init__(self, images, thresh):

		self.images = images
		self.thresh = thresh

	def distancemap(self):
		'''
		This function is to generate distance map of the thresh image. We use the opencv
		function distanceTransform to generate it. Moreover, in this case, we use Euclidiean
		Distance (DIST_L2) as a metric of distance. 

		Input: None

		Output: Image distance map

		'''
		return [cv2.distanceTransform(self.thresh[i], distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)\
  				for i in range(len(self.thresh))]

	def new_image(self, alpha, dismap):
		'''
		This function is to generate a new image combining the oringal image I0 with
		the distance map image Idis by following expression:
								Inew = I0 + alpha*Idis
		In this program, we choose alpha as 0.4.

		Input: the weight of distance map: alpha
			   the distance map image

		Output: new grayscale image

		'''
		
		return [self.images[i] + alpha * dismap[i] for i in range(len(self.thresh))]

	def compute_gvf(self, newimage):
		'''
		This function is to compute the gradient vector of the imput image.

		Input: a grayscale image with size, say m * n * # of images

		Output: a 3 dimentional image with size, m * n * 2, where the last dimention is
		the gradient vector (gx, gy)

		'''
		kernel_size = 5 # kernel size for blur image before compute gradient
		newimage = [cv2.GaussianBlur((np.clip(newimage[i], 0, 255)).astype(np.uint8),(kernel_size,kernel_size),0)\
		            for i in range(len(self.thresh))]
		# use sobel operator to compute gradient 
		temp = np.zeros((newimage[0].shape[0], newimage[0].shape[1], 2), np.float32) # store temp gradient image 
		gradimg = []  # output gradient images (height * weight * # of images)
		
		for i in range(len(newimage)):
			# compute sobel operation in x, y directions
			gradx = cv2.Sobel(newimage[i],cv2.CV_64F,1,0,ksize=3)
  			grady = cv2.Sobel(newimage[i],cv2.CV_64F,0,1,ksize=3)
  			# add the gradient vector
  			temp[:,:,0], temp[:,:,1] = gradx, grady
			gradimg.append(temp)

		return gradimg

		
	def find_certer(self, gvfimage, index):
		'''
		This function is to find the center of Nuclei.

		Input: the gradient vector image (height * weight * 2).

		Output: the record image height * weight).

		'''
		# Initialize a image to record seed candidates.
		imgpair = np.zeros(gvfimage.shape[:2])

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
		erthresh = cv2.erode(self.thresh[index].copy(), kernel, iterations = 1)
		while erthresh.sum() > 0:

			print "how many left? ", erthresh.sum()
			# Initialize partical coordinates [y, x]
			y0, x0 = np.where(erthresh>0)
			p0 = np.array([y0[0], x0[0], 1])

			# Initialize record coordicates [y, x]
			p1 = np.array([5000, 5000, 1])

			# mark the first non-zero point of thresh image to 0
			erthresh[p0[0], p0[1]] = 0

			# a variable to record if the point out of bound of image or 
			# out of maximum loop times
			outbound = False

			# count loop times to limit max loop times
			count = 0

			while sp.distance.cdist([p0],[p1]) > 1:

				count += 1
				p1 = p0
				u = gvfimage[p0[0], p0[1], 1]
				v = gvfimage[p0[0], p0[1], 0]
				M = np.array([[1, 0, u],\
				              [0, 1, v],\
				              [0, 0, 1]], np.float32)
				p0 = M.dot(p0)
				if not inbounds(self.thresh[index].shape, (p0[0], p0[1])) or count > looplimit:
					outbound = True
					if count > looplimit:
						print "  count > looplimit..."
					break

			if not outbound:
				imgpair[p0[0], p0[1]] += 1

		imgpair_raw = imgpair.copy()

		# find local maximum in a certain range in order to remove multiple
		# point in one necleus.
		neighborhood_size = 20
		data_max = ndimage.filters.maximum_filter(imgpair, neighborhood_size)
		data_max[data_max==0] = 255
		imgpair = (imgpair == data_max)

		# in order to remove the points that has same value in one necleus that 
		# cannot be remove by previous step, we measure the distance between 
		# each point, and get rid of the points that to close to the other. 
		binary_imgpair_raw = imgpair.copy()
		binary_imgpair_raw = binary_imgpair_raw.astype(np.uint8)
		binary_imgpair_raw[binary_imgpair_raw>0] = 255
		y, x = np.where(imgpair>0)
		points = zip(y[:], x[:])
		dmap = distance.cdist(points, points, 'euclidean')
		y, x = np.where(dmap<20)
		ps = zip(y[:], x[:])
		for p in ps:
			if p[0] != p[1]:
				imgpair[points[min(p[0], p[1])]] = 0

		return imgpair.astype(np.uint8)*255, binary_imgpair_raw, imgpair_raw

def main():
	'''
	This part is for testing gvf.py with single image.

	Input: an original image

	Output: Thresholding image and seed image

	'''
	images = []
	temp = cv2.imread(sys.argv[1])
	images.append(cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY))
	# Binarization
	th = athresh(images)
	threh = th.applythresh()
	# Nuclei center detection
	gvf = GVF(images, threh)
	dismap = gvf.distancemap()
	newimg = gvf.new_image(0.4, dismap) # choose alpha as 0.4.
	gradimg = gvf.compute_gvf(newimg)

	imgpair = gvf.find_certer(gradimg[0], 0)
	cv2.imwrite('imgpair_test.tif', (np.clip(imgpair, 0, 255)).astype(np.uint8))

# if python says run, then we should run
if __name__ == '__main__':
    main()
