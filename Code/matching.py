'''
This file is to match nuclei in two consecutive frames by Phase Controlled Optimal Matching. 
It includes two part: 
		1) Dissimilarity measure
		2) Matching

'''
import cv2
import numpy as np
import sys
from graph_construction import GRAPH
import os
import imageio

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

Max_dis = 100000

def write_image(image, title, index, imgformat='.tif'):
    if index < 10:
            name = '00'+str(index)
    else:
        name = '0'+str(index)
    cv2.imwrite(title+name+imgformat, image.astype(np.uint16))

class FEAVECTOR():
	'''
	This class builds a feature vector for each segments.
	The format of each vector is: 
					v(k,i) = [c(k,i), s(k, i), h(k, i), e(k, i)], where k is the 
					index of the image (frame) and i is the label of each segment. 

					c(k,i): the centroid of each segment (y, x);
					s(k,i): the binary shape of each segment;
					h(k,i): the intensity distribution (hsitogram) of the segment;
					e(k,i): the spatial distribution of the segment. Its format is 
					like (l(k, i, p), theta(k, i, p)), where p represent different
					line connected with different segment. 

	'''
	def __init__(self, centroid=None, shape=None, histogram=None, spatial=None, \
		               ID=None, start = None, end=None, label=None):
		self.c = centroid
		self.s = shape
		self.h = histogram
		self.e = spatial
		self.id = ID
		self.start = start
		self.end = end
		self.l = label

	def add_id(self, num, index):
		'''
		This function adds cell id for each cell.

		'''
		if index == 0:
			self.id = np.linspace(1, num, num)
		else:
			self.id= np.linspace(-1, -1, num)

	def add_label(self):
		'''
		This function is to add labels for each neclei for post process.

		'''
		self.l = np.linspace(0, 0, len(self.c))

	def set_centroid(self, centroid):
		'''
		This function sets the centroid for all neclei.

		Input: the set of centroid: # of images * # of neclei * 2 (y, x)

		Output: None

		'''
		self.c = centroid

	def set_spatial(self, spatial):
		'''
		This function sets the spatial distrbution for all neclei.

		Input: the set of centroid: # of images * # of neclei * # of line segments (length, slope)

		Output: None

		'''
		self.e = spatial

	def set_shape(self, image, marker):
		'''
		This function sets the binary shape for all necluei.

		Input: the original images: # of images * height * weight
		       the labeled images: # of images * nucei's height * nucei's weight ()

		Output: None

		'''
		def boundingbox(image):
 			y, x = np.where(image)
			return min(x), min(y), max(x), max(y)

		shape = []

		for label in range(1, marker.max()+1):
			tempimg = marker.copy()
			tempimg[tempimg!=label] = 0
			tempimg[tempimg==label] = 1
			if tempimg.sum():
				minx, miny, maxx, maxy = boundingbox(tempimg)
				shape.append((tempimg[miny:maxy+1, minx:maxx+1], image[miny:maxy+1, minx:maxx+1]))
			else:
				shape.append(([], []))

		self.s = shape

	def set_histogram(self):
		'''
		Note: this function must be implemneted after set_shape().

		'''
		def computehistogram(image):
			h, w = image.shape[:2]
			his = np.zeros((256,1))
			for y in range(h):
				for x in range(w):
					his[image[y, x], 0] += 1
			return his

		assert self.s != None, "this function must be implemneted after set_shape()."

		his = []

		for j in range(len(self.s)):
			img = self.s[j][1]
			if len(img):
				temphis = computehistogram(img)
				his.append(temphis)
			else:
				his.append(np.zeros((256,1)))

		self.h = his

	def generate_vector(self):
		'''
		This function is to convert the vector maxtrics into a list.

		Output: a list of vector: [v0, v1, ....]

		'''
		vector = []
		for i in range(len(self.c)):
			vector.append(FEAVECTOR(centroid=self.c[i], \
				                    shape=self.s[i], \
				                    histogram=self.h[i], \
				                    spatial=self.e[i], \
				                    ID=self.id[i], \
				                    label=self.l[i]))
		return vector

class SIMPLE_MATCH():
	'''
	This class is simple matching a nucleus into a nucleus in the previous frame by 
	find the nearest neighborhood. 

	'''
	def __init__(self, index0, index1, images, vectors):
		self.v0 = vectors[index0]
		self.v1 = vectors[index1]
		self.i0 = index0
		self.i1 = index1
		self.images = images
		self.vs = vectors

	def distance_measure(self, pv0, pv1, alpha1=0.31, alpha2=0.15, alpha3=0.23, alpha4=0.31, phase = 1):
		'''
		This function measures the distence of the two given feature vectors. 

		This distance metrics we use is:
						d(v(k, i), v(k+1, j)) = alpha1 * d(c(k, i), c(k+1, j)) + 
												alpha2 * q1 * d(s(k, i), s(k+1, j)) +
												alpha3 * q2 * d(h(k, i), h(k+1, j)) +
												alpha4 * d(e(k, i), e(k+1, j))
		Input: The two given feature vectors, 
		       and the set of parameters.

		Output: the distance of the two given vectors. 

		'''
		def centriod_distance(c0, c1, D=30.):
			dist = np.sqrt((c0[0]-c1[0])**2 + (c0[1]-c1[1])**2)
			return dist/D if dist < D else 1

		def maxsize_image(image1, image2):
 			y1, x1 = np.where(image1)
 			y2, x2 = np.where(image2)
			return min(min(x1), min(x2)), min(min(y1), min(y2)), \
			       max(max(x1), max(x2)), max(max(y1), max(y2)),

		def symmetry(image, shape):
			h, w = image.shape[:2] 
			newimg = np.zeros(shape)
			newimg[:h, :w] = image
			v = float(shape[0] - h)/2.
			u = float(shape[1] - w)/2.
			M = np.float32([[1,0,u],[0,1,v]])
			return cv2.warpAffine(newimg,M,(shape[1],shape[0]))

		def shape_distance(s0, s1):
			minx, miny, maxx, maxy = maxsize_image(s0, s1)
			height = maxy - miny + 1
			width = maxx - minx + 1

			img0 = symmetry(s0, (height, width))
			img1 = symmetry(s1, (height, width))

			num = 0.
			deno = 0.
			for y in range(height):
				for x in range(width):
					if img0[y, x] and img1[y, x]:
						num += 1
					if img0[y, x] or img1[y, x]:
						deno += 1

			return 1 - num/deno

		def histogram_distance(h0, h1): 
			max_p = max([img.max() for img in self.images])
			min_p = min([img.min() for img in self.images])
			
			num = 0.
			deno = 0.
			for t in range(min_p, max_p+1):
				num += abs(h0[t] - h1[t])
				deno += max(h0[t], h1[t])

			return num/deno

		def spatial_distance(e0, e1):
			m = float(len(e0))
			n = float(len(e1))
			if not n or not m:
				return Max_dis
			sumdis = 0.
			for pe1 in e1:
				dis = None
				for pe0 in e0:
					tempdis = abs(pe0[0]-pe1[0])/max(pe0[0], pe1[0]) * \
					                                           abs(pe0[1]-pe1[1])
					if dis == None:
						dis = tempdis
					elif dis > tempdis:
						dis = tempdis
				sumdis += dis
			return sumdis*(1./n)

		if len(pv0.s[0]) and len(pv1.s[0]):
			dist = 	alpha1 * centriod_distance(pv0.c, pv1.c)+ \
					alpha2 * shape_distance(pv0.s[1], pv1.s[1]) * phase + \
					alpha3 * histogram_distance(pv0.h, pv1.h) * phase + \
					alpha4 * spatial_distance(pv0.e, pv1.e)
		else:
			dist = Max_dis

		return dist

	def phase_identify(self, pv1, min_times_MA2ma = 2):
			_, contours, hierarchy = cv2.findContours(pv1.s[0].astype(np.uint8), 1, 2)
			if not len(contours):
				return 1
			cnt = contours[0]
			if len(cnt) >= 5:
				(x,y),(ma,MA),angle = cv2.fitEllipse(cnt)
				if ma and MA/ma > min_times_MA2ma:
					return 0
				elif not ma and MA:
					return 0
				else:
					return 1
			else:
				return 1

	def find_match(self, max_distance=1):
		'''
		This function is to find the nearest neighborhood between two
		successive frame.

		'''
		def centriod_distance(c0, c1, D=30.):
			dist = np.sqrt((c0[0]-c1[0])**2 + (c0[1]-c1[1])**2)
			return dist/D if dist < D else 1

		for i, pv1 in enumerate(self.v1):
			dist = np.ones((len(self.v0), 3), np.float32)
			count = 0
			q = self.phase_identify(pv1, 3)
			for j, pv0 in enumerate(self.v0):
				if centriod_distance(pv0.c, pv1.c) < 1:
					dist[count][0] = self.distance_measure(pv0, pv1, phase=q)
					dist[count][1] = pv0.l
					dist[count][2] = pv0.id
					count += 1
			sort_dist = sorted(dist, key=lambda a_entry: a_entry[0]) 
			if sort_dist[0][0] < max_distance:
				self.v1[i].l = sort_dist[0][1]
				self.v1[i].id = sort_dist[0][2]

	def mitosis_refine(self):
		'''
		This function is to find died cell due to the by mitosis. 

		'''
		def find_sibling(pv0):
			'''
			This function is to find sibling cells according to the centroid of
			pv0. The criteria of sibling is:
					1. the jaccard cooeficient of the two cells is above 0.5
					2. the sum of the two areas should in the range [A, 2.5A], where
					   A is the area of the pv0
					3. the position of the two cells should be not larger than 20 pixels.

			Input: pv0: the parent cell that you want to find siblings;

			Output: the index of the siblings.

			'''
			def maxsize_image(image1, image2):
	 			y1, x1 = np.where(image1)
	 			y2, x2 = np.where(image2)
				return min(min(x1), min(x2)), min(min(y1), min(y2)), \
				       max(max(x1), max(x2)), max(max(y1), max(y2)),

			def symmetry(image, shape):
				h, w = image.shape[:2] 
				newimg = np.zeros(shape)
				newimg[:h, :w] = image
				v = float(shape[0] - h)/2.
				u = float(shape[1] - w)/2.
				M = np.float32([[1,0,u],[0,1,v]])
				return cv2.warpAffine(newimg,M,(shape[1],shape[0]))

			def jaccard(s0, s1):
				minx, miny, maxx, maxy = maxsize_image(s0, s1)
				height = maxy - miny + 1
				width = maxx - minx + 1

				img0 = symmetry(s0, (height, width))
				img1 = symmetry(s1, (height, width))

				num = 0.
				deno = 0.
				for y in range(height):
					for x in range(width):
						if img0[y, x] and img1[y, x]:
							num += 1
						if img0[y, x] or img1[y, x]:
							deno += 1

				return num/deno

			sibling_cand = []
			for i, pv1 in enumerate(self.v1): 
				if np.linalg.norm(pv1.c-pv0.c) < 50:
					sibling_cand.append([pv1, i])

			sibling_pair = []
			area = pv0.s[0].sum()
			jaccard_value = []
			for sibling0 in sibling_cand:
				for sibling1 in sibling_cand:
					if (sibling1[0].c != sibling0[0].c).all():
						sum_area = sibling1[0].s[0].sum()+sibling0[0].s[0].sum()
						similarity = jaccard(sibling0[0].s[0], sibling1[0].s[0])
						if similarity > 0.4 and (sum_area > 2*area):
							sibling_pair.append([sibling0, sibling1])
							jaccard_value.append(similarity)
			if len(jaccard_value):
				return sibling_pair[np.argmax(jaccard_value)]
			else:
				return 0

		v1_ids = []
		for pv1 in self.v1:
			v1_ids.append(pv1.id)

		for pv0 in self.v0:
			if pv0.id not in v1_ids and len(pv0.s[0]) and not self.phase_identify(pv0):
				sibling = find_sibling(pv0)
				if sibling:
					[s0, s1] = sibling
					if s0[0].l==0 and s1[0].l==0 and \
					   s0[0].id==-1 and s1[0].id==-1:
						self.v1[s0[1]].l = pv0.id
						self.v1[s1[1]].l = pv0.id

		return self.v1

	def match_missing(self, mask, max_frame = 1, max_distance = 10, min_shape_similarity = 0.6):
		'''
		This function is to match the cells that didn't show in the last frame caused by 
		program fault. In order to match them, we need to seach the cell in the previous 
		frame with in the certain range and with similar shape.  

		'''
		def centriod_distance(c0, c1):
			dist = np.sqrt((c0[0]-c1[0])**2 + (c0[1]-c1[1])**2)
			return dist

		def maxsize_image(image1, image2):
 			y1, x1 = np.where(image1)
 			y2, x2 = np.where(image2)
			return min(min(x1), min(x2)), min(min(y1), min(y2)), \
			       max(max(x1), max(x2)), max(max(y1), max(y2)),

		def symmetry(image, shape):
			h, w = image.shape[:2] 
			newimg = np.zeros(shape)
			newimg[:h, :w] = image
			v = float(shape[0] - h)/2.
			u = float(shape[1] - w)/2.
			M = np.float32([[1,0,u],[0,1,v]])
			return cv2.warpAffine(newimg,M,(shape[1],shape[0]))

		def shape_similarity(s0, s1):
			if len(s0) and len(s1):
				minx, miny, maxx, maxy = maxsize_image(s0, s1)
				height = maxy - miny + 1
				width = maxx - minx + 1

				img0 = symmetry(s0, (height, width))
				img1 = symmetry(s1, (height, width))

				num = 0.
				deno = 0.
				for y in range(height):
					for x in range(width):
						if img0[y, x] and img1[y, x]:
							num += 1
						if img0[y, x] or img1[y, x]:
							deno += 1
				return num/deno

			else:
				return 0.

		def add_marker(index_find, index_new, pv0_id):
			temp = mask[index_new]
			find = mask[index_find]
			temp[find==pv0_id] = pv0_id
			print "index_new: ", index_new
			return temp

		for i, pv1 in enumerate(self.v1):
			if pv1.id == -1:
				for index in range(1, max_frame+1):
					if self.i0-index >= 0:
						vector = self.vs[self.i0-index]
						for pv0 in vector:
							if centriod_distance(pv0.c, pv1.c) < max_distance and \
								shape_similarity(pv0.s[0], pv1.s[0]) > min_shape_similarity:
								self.v1[i].id = pv0.id
								self.v1[i].l = pv0.l
								print "missing in frame: ", self.i1, "find in frame: ", \
								       self.i0-index, "ID: ", pv0.id, " at: ", pv0.c
								for i in range(self.i0-index+1, self.i1):
									mask[i] = add_marker(self.i0-index, i, pv0.id)
		return mask

	def new_id(self):
		'''
		This function is to add new labels for the necles that are marked as -1.


		'''
		def find_max_id(vectors):
			max_id = 0
			for vector in vectors:
				for pt in vector:
					if pt.id > max_id:
						max_id = pt.id 
			return max_id
		max_id = find_max_id(self.vs)
		max_id += 1
		for i, pv1 in enumerate(self.v1):
			if pv1.id == -1:
				self.v1[i].id = max_id
				max_id += 1

	def generate_mask(self, marker, index, isfinal=False):
		'''
		This function is to generate a 16-bit image as mask image. 

		'''
		h, w = marker.shape[:2]
		mask = np.zeros((h, w), np.uint16)
		pts = list(set(marker[marker>0]))
		if not isfinal:
			assert len(pts)==len(self.v0), 'len(pts): %s != len(self.v0): %s' % (len(pts), len(self.v0))
			for pt, pv in zip(pts, self.v0):
				mask[marker==pt] = pv.id

		else:
			assert len(pts)==len(self.v1), 'len(pts): %s != len(self.v0): %s' % (len(pts), len(self.v1))
			for pt, pv in zip(pts, self.v1):
				mask[marker==pt] = pv.id

		os.chdir("test")
		write_image(mask, "mask", index)
		os.chdir(os.pardir)
		return mask	

	def return_vectors(self):
		'''
		This function is to return the vectors that we have already 
		changed.

		Output: the vectors from the k+1 frame.

		'''	
		return self.v1

def set_date(vectors):
	'''
	This function is to add the start and end frame of each vector and
	combine the vector with same id.

	Input: the list of vectors in different frames. 

	Output: the list of vectors of all cell with different id. 

	'''
	max_id = 0
	for vector in vectors:
		for pv in vector:
			if pv.id > max_id:
				max_id = pv.id

	print "max_id: ", max_id
	output = np.zeros((max_id, 4))
	output[:,0] = np.linspace(1, max_id, max_id) # set the cell ID
	output[:,1] = len(vectors)
	for frame, vector in enumerate(vectors):
		for pv in vector:
			if output[pv.id-1][1] > frame:     # set the start frame
				output[pv.id-1][1] = frame
			if output[pv.id-1][2] < frame:     # set the end frame
				output[pv.id-1][2] = frame
			output[pv.id-1][3] = pv.l          # set tht cell parent ID

	return output

def write_info(vector, name):
	'''
	This function is to write info. of each vector.

	Input: the list of vector generated by set_date() and 
	       the name of output file.

	'''
	with open(name+".txt", "w+") as file:
		for p in vector:
			file.write(str(int(p[0]))+" "+\
				       str(int(p[1]))+" "+\
				       str(int(p[2]))+" "+\
				       str(int(p[3]))+"\n")

'''
This part is to test the matching scheme with single image

Input: the original image;
       the labeled image;
       the binary labeled image.

'''
def main():
	def normalize(image):
		'''
		This function is to normalize the input grayscale image by
		substracting globle mean and dividing standard diviation for
		visualization. 

		Input:  a grayscale image

		Output: normolized grascale image

		'''
		img = image.copy().astype(np.float32)
		img -= np.mean(img)
		img /= np.linalg.norm(img)
		img = (img - img.min() )
		img *= (1./float(img.max()))
		return (img*255).astype(np.uint8)

	path=os.path.join("resource/training/01")
	for r,d,f in os.walk(path):
		images = []
		enhance_images = []
		for files in f:
			if files[-3:].lower()=='tif':
				temp = cv2.imread(os.path.join(r,files))
				gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY) 
				images.append(gray.copy())
				enhance_images.append(normalize(gray.copy()))

	print "Total number of image is ", len(images)
	print "The shape of image is ", images[0].shape, type(images[0][0,0])

	path=os.path.join("test/mark_train_01")
	markers = []
	for r,d,f in os.walk(path):
		for files in f:
			if files[:1].lower()=='m':
				temp = cv2.imread(os.path.join(r,files))
				gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY) 
				markers.append(gray.copy())

	path=os.path.join("test/binarymark_train_01")
	binarymark = []
	for r,d,f in os.walk(path):
		for files in f:		
			if files[:1].lower()=='b':
				temp = cv2.imread(os.path.join(r,files))
				gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY) 
				binarymark.append(gray.copy())

	print "Total number of markers is ", len(markers)
	print "The shape of markers is ", markers[0].shape, type(markers[0][0,0])

	print "Total number of binarymark is ", len(binarymark)
	print "The shape of binarymark is ", binarymark[0].shape, type(binarymark[0][0,0])

	centroid = []
	slope_length = []

	# Build Delaunay Triangulation
	for i in range(len(images)):
		print "  graph_construction: image ", i
		graph = GRAPH(markers, binarymark, i)
		tempcentroid, tempslope_length = graph.run()
		centroid.append(tempcentroid)
		slope_length.append(tempslope_length)

	# Feature vector construction
	vector = []
	max_id = 0
	mask = []
	for i in range(len(images)):
		print "  feature vector: image ", i
		v = FEAVECTOR()
		v.set_centroid(centroid[i])
		v.set_spatial(slope_length[i])
		v.set_shape(images[i], markers[i])
		v.set_histogram()
		v.add_label()
		v.add_id(markers[i].max(), i)
		vector.append(v.generate_vector())
		
		print "num of nuclei: ", len(vector[i])

	# Feature matching
	for i in range(len(images)-1):
		print "  Feature matching: image ", i
		m = SIMPLE_MATCH(i,i+1,[images[i], images[i+1]], vector)
		mask.append(m.generate_mask(markers[i], i))	
		m.find_match(0.3)
		mask = m.match_missing(mask, max_frame=2, max_distance=20)
		vector[i+1] = m.mitosis_refine()
		m.new_id()	
		vector[i+1] = m.return_vectors()

	print "  Feature matching: image ", i+1
	mask.append(m.generate_mask(markers[i+1], i+1, True))

	cells = set_date(vector)
	write_info(cells, "res_track")
	
	def find_max_id(vector):
		max_id = 0
		for pv in vector:
			for p in pv:
				if p.id > max_id:
					max_id = p.id
		return max_id

	# This part is to mark the result in the normolized image and 
	# write the gif image.
	max_id = find_max_id(vector)
	colors = [np.random.randint(0, 255, size=max_id),\
	          np.random.randint(0, 255, size=max_id),\
	          np.random.randint(0, 255, size=max_id)]
	font = cv2.FONT_HERSHEY_SIMPLEX 
	selecy_id = 9
	for i, m in enumerate(mask):
		print "  write the gif image: image ", i
		enhance_images[i] = cv2.cvtColor(enhance_images[i],cv2.COLOR_GRAY2RGB)
		for pv in vector[i]:
			center = pv.c
			if pv.l == selecy_id or pv.id == selecy_id:
				if not pv.l:
					color = (colors[0][int(pv.id)-1],\
					         colors[1][int(pv.id)-1],\
					         colors[2][int(pv.id)-1],)
				else:
					color = (colors[0][int(pv.l)-1],\
					         colors[1][int(pv.l)-1],\
					         colors[2][int(pv.l)-1],)

				if m[center[0], center[1]]:
					enhance_images[i][m==pv.id] = color
					cv2.putText(enhance_images[i],\
						        str(int(pv.id)),(int(pv.c[1]), \
						        int(pv.c[0])), 
						        font, 0.5,\
						        (255,255,255),1)
	imageio.mimsave('mitosis_final.gif', enhance_images, duration=0.6)

	# This part is for showing the result in 3D plot. 
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	colors = cm.rainbow(np.linspace(0, 1, max_id))
	colors = np.random.permutation(colors)
	ax.set_xlim(0, images[0].shape[0])
	ax.set_ylim(0, images[0].shape[1])
	ax.set_zlim(0, len(images))
	ax.set_xlabel('x axis')
	ax.set_ylabel('y axis')
	ax.set_zlabel('frame')
	plotimg = []

	def scatter_pts(vector, ax, i):
		for pt in vector:
			if not len(pt.s[0]) or sum(pt.c) == 0:
				continue

			x_p = pt.c[0]
			y_p = pt.c[1]
			z_p = i
			if pt.l == selecy_id or (pt.id == selecy_id and pt.l==0):
				if not pt.l:
					ax.scatter(x_p, y_p, z_p, s=2, color=colors[int(pt.id)-1])
				else:
					ax.scatter(x_p, y_p, z_p, s=2, color=colors[int(pt.l)-1])

	scatter_pts(vector[1], ax, 0)
	plt.savefig('testplot.jpg', dpi=fig.dpi)
	plotimg.append(cv2.imread('testplot.jpg'))
	
	for i in range(len(images)-1):
		print "  Feature plot: image ", i

		for pt1 in vector[i+1]:
			for pt0 in vector[i]:
				if (pt0.id == pt1.id or pt0.id == pt1.l) and \
				   (pt1.l == selecy_id or (pt1.id == selecy_id and pt1.l==0)):
					x1 = pt1.c[0]
					x0 = pt0.c[0]
					y1 = pt1.c[1]
					y0 = pt0.c[1]
					z1 = (i+1)
					z0 = i
					if not pt0.l:
						ax.plot([x1, x0], \
							    [y1, y0], \
							    [z1, z0], \
							    color=colors[int(pt0.id)-1])
					else:
						ax.plot([x1, x0], \
							    [y1, y0], \
							    [z1, z0], \
							    color=colors[int(pt0.l)-1])
		plt.savefig('testplot.jpg', dpi=fig.dpi)
		plotimg.append(cv2.imread('testplot.jpg'))

	plt.show()
	imageio.mimsave('temporary_result/plotimg.gif', plotimg, duration=0.6)
	

# if python says run, then we should run
if __name__ == '__main__':
    main()
				