"""Make faded plot numbers black-on-white: contrast against local brightest neighbour."""
import sys, numpy as np, cv2
from PIL import Image
im = np.array(Image.open(sys.argv[1]).convert('RGB'))
g = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)[:,:,0].astype(np.float32)
bright = cv2.dilate(g, np.ones((9,9),np.uint8))
c = np.clip((bright - g) * 5.0, 0, 255)
Image.fromarray((255 - c).astype(np.uint8)).save(sys.argv[2])
