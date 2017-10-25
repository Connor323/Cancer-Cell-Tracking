import imageio
import cv2
import glob
from PIL import Image
import numpy as np
import os

files = glob.glob("images/*.gif")
for f in files:
	img_out = []
	img = Image.open(f)
	try:
		while 1:
		        img.seek(img.tell()+1)
		        rgb_im = img.convert('RGB')
		        tmp = np.array(rgb_im)
		        h, w = tmp.shape[:2]
		        tmp = cv2.resize(tmp, (w/2, h/2), interpolation=cv2.INTER_CUBIC)
		        img_out.append(tmp)
	except EOFError:
		img_out = np.array(img_out)
		print img_out.shape
		imageio.mimsave(os.path.basename(f), img_out)
