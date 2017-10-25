'''
This file is to generate a neighboring graph contraction using 
Delaunary Triangulation.

'''

import cv2
import numpy as np
import random
import sys
import imageio
 
class GRAPH():
    '''
    This class contains all the functions needed to compute 
    Delaunary Triangulation.

    '''
    def __init__(self, mark, binary, index):
        '''
        Input: the grayscale mark image with different label on each segments
               the binary image of the mark image
               the index of the image

        '''
        self.mark = mark[index]
        self.binary = binary[index]
              
    def rect_contains(self, rect, point):
        '''
        Check if a point is inside the image

        Input: the size of the image 
               the point that want to test

        Output: if the point is inside the image

        '''
        if point[0] < rect[0] :
            return False
        elif point[1] < rect[1] :
            return False
        elif point[0] > rect[2] :
            return False
        elif point[1] > rect[3] :
            return False
        return True
     
    def draw_point(self, img, p, color ):
        '''
        Draw a point

        '''
        cv2.circle( img, (p[1], p[0]), 2, color, cv2.FILLED, 16, 0 )
     
    def draw_delaunay(self, img, subdiv, delaunay_color ):
        '''
        Draw delaunay triangles and store these lines

        Input: the image want to draw
               the set of points: format as cv2.Subdiv2D
               the color want to use

        Output: the slope and length of each line ()

        '''
        triangleList = subdiv.getTriangleList();
        size = img.shape
        r = (0, 0, size[0], size[1])

        slope_length = [[]]
        for i in range(self.mark.max()-1):
            slope_length.append([])

        for t in triangleList:
             
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
             
            if self.rect_contains(r, pt1) and self.rect_contains(r, pt2) and self.rect_contains(r, pt3):
                
                # draw lines
                cv2.line(img, (pt1[1], pt1[0]), (pt2[1], pt2[0]), delaunay_color, 1, 16, 0)
                cv2.line(img, (pt2[1], pt2[0]), (pt3[1], pt3[0]), delaunay_color, 1, 16, 0)
                cv2.line(img, (pt3[1], pt3[0]), (pt1[1], pt1[0]), delaunay_color, 1, 16, 0)

                # store the length of line segments and their slopes
                for p0 in [pt1, pt2, pt3]:
                    for p1 in [pt1, pt2, pt3]:
                        if p0 != p1:
                            temp = self.length_slope(p0, p1)
                            if temp not in slope_length[self.mark[p0]-1]:
                                slope_length[self.mark[p0]-1].append(temp)

        return slope_length

    def length_slope(self, p0, p1):
        '''
        This function is to compute the length and theta for the given two points.

        Input: two points with the format (y, x)

        '''
        if p1[1]-p0[1]:
            slope = (p1[0]-p0[0]) / (p1[1]-p0[1])
        else:
            slope = 1e10

        length = np.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)

        return length, slope

    def generate_points(self):
        '''
        Find the centroid of each segmentation

        '''
        centroids = []
        label = []
        max_label = self.mark.max()

        for i in range(1, max_label+1):
            img = self.mark.copy()
            img[img!=i] = 0
            if img.sum():
                _, contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
                m = cv2.moments(contours[0])

                if m['m00']:
                    label.append(i)
                    centroids.append(( int(round(m['m01']/m['m00'])),\
                                       int(round(m['m10']/m['m00'])) ))
                else:
                    label.append(i)
                    centroids.append(( 0,0 ))

        return centroids, label

    def run(self, animate = False):
        '''
        The pipline of graph construction.

        Input: if showing a animation (False for default)

        Output: centroids: # of segments * 2   (y, x)
                slopes and length: # of segments * # of slope_length

        '''
        # Read in the image.
        img_orig = self.binary.copy()
         
        # Rectangle to be used with Subdiv2D
        size = img_orig.shape
        rect = (0, 0, size[0], size[1])
         
        # Create an instance of Subdiv2D
        subdiv = cv2.Subdiv2D(rect);
        
        # find the centroid of each segments
        points, label = self.generate_points()


        # add and sort the centroid to a numpy array for post processing
        centroid = np.zeros((self.mark.max(), 2))
        for p, l in zip(points, label):
            centroid[l-1] = p

        outimg = []
        # Insert points into subdiv
        for p in points:
            subdiv.insert(p)
             
            # Show animation
            if animate:
                img_copy = img_orig.copy()
                # Draw delaunay triangles
                self.draw_delaunay( img_copy, subdiv, (255, 255, 255) );
                outimg.append(img_copy)
                cv2.imshow("win_delaunay", img_copy)
                cv2.waitKey(50)

        imageio.mimsave('graph_contruction.gif', outimg, duration=0.3)
        # Draw delaunay triangles
        slope_length = self.draw_delaunay( img_orig, subdiv, (255, 255, 255) );
     
        # Draw points
        for p in points :
            self.draw_point(img_orig, p, (0,0,255))
        
        # show images
        if animate:
            cv2.imshow('img_orig',img_orig)
            k = cv2.waitKey(0)
            if k == 27:         # wait for ESC key to exit
                cv2.destroyAllWindows()
            elif k == ord('s'): # wait for 's' key to save and exit
                cv2.imwrite('messigray.png',img)
                cv2.destroyAllWindows()

        return centroid, slope_length

'''
This part is the small test for graph_contruction.py.

Input:  grayscale marker image
        binary marker image

Output: a text file includes the centroid and the length and slope for each neighbor. 

'''
def main():
    mark = [cv2.imread(sys.argv[1])]
    binary = [cv2.imread(sys.argv[2])]
    mark[0] = cv2.cvtColor(mark[0], cv2.COLOR_BGR2GRAY)
    binary[0] = cv2.cvtColor(binary[0], cv2.COLOR_BGR2GRAY)
    graph = GRAPH(mark, binary, 0)
    centroid, slope_length = graph.run(True)
    with open("centroid_slope_length.txt", "w+") as file:
        for i, p in enumerate(centroid):
            file.write(str(p[0])+" "+str(p[1])+"     "+str(slope_length[i])+"\n")

# if python says run, then we should run
if __name__ == '__main__':
    main()