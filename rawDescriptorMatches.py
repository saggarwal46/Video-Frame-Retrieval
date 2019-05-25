import numpy as np
import scipy.io
import glob
from scipy import misc
import matplotlib.pyplot as plt
from displaySIFTPatches import displaySIFTPatches
from selectRegion import roipoly
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
from skimage.color import rgb2gray
from dist2 import dist2
import matplotlib.cm as cm
import pylab as pl
import pdb

framesdir = 'twoFrameData.mat'
mat = scipy.io.loadmat(framesdir)
img1 = mat['im1']
img2 = mat['im2']

pl.imshow(img1)
MyROI = roipoly(roicolor='r')
Ind = MyROI.getIdx(img1, mat['positions1'])
computedDist = dist2(mat['descriptors1'][Ind,:], mat['descriptors2'])
Ind = []
for row in computedDist:
    Ind.append(np.argmin(row))

fig=plt.figure()
ax = fig.add_subplot(121)
ax.imshow(img1)
MyROI.fig= fig
MyROI.ax=ax
MyROI.displayROI()
bx=fig.add_subplot(122)
bx.imshow(img2)
coners = displaySIFTPatches(mat['positions2'][Ind,:], mat['scales2'][Ind,:], mat['orients2'][Ind,:])
for j in range(len(coners)):
    bx.plot([coners[j][0][1], coners[j][1][1]], [coners[j][0][0], coners[j][1][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners[j][1][1], coners[j][2][1]], [coners[j][1][0], coners[j][2][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners[j][2][1], coners[j][3][1]], [coners[j][2][0], coners[j][3][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners[j][3][1], coners[j][0][1]], [coners[j][3][0], coners[j][0][0]], color='g', linestyle='-', linewidth=1)
bx.set_xlim(0, img1.shape[1])
bx.set_ylim(0, img1.shape[0])
plt.gca().invert_yaxis()

plt.show()
