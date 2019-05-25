import numpy as np
import glob
import scipy.io
from scipy import misc
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from displaySIFTPatches import displaySIFTPatches
from selectRegion import roipoly
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
from skimage.color import rgb2gray
from dist2 import dist2
import matplotlib.cm as cm
import pylab as pl
import pdb
import pickle
import random
import heapq
import warnings
from numpy.linalg import norm


def bag_of_words_histogram(vocabulary, descriptors, k):
    hist = [0] * k
    mappings = vocabulary.predict(descriptors)
    for map in mappings:
        hist[map] += 1
    return hist

def normalizedScalarProduct(histA, histB, k):
    score = np.dot(histA, histB) / norm(histA, ord=2) * norm(histB, ord=2)
    return score

vocabSize = 1500
filename = 'finalized_model_' + str(vocabSize) + '.sav'
tfilename = 'trainging' + str(vocabSize)
hfilename = 'hists' + str(vocabSize) + '.npy'
kmeansminbatch = pickle.load(open(filename, 'rb'))
sample_images = pickle.load(open(tfilename, 'rb'))
framesdir = 'frames/'
siftdir = 'sift/'
fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]
sample_images = pickle.load(open(tfilename, 'rb'))
M = 5

query_idx = [864]


try:
    histograms = np.load(hfilename)
except:
    histograms = np.empty((0, vocabSize))
    for i in sample_images:
        fname = siftdir + fnames[i]
        try:
            mat = scipy.io.loadmat(fname)
            descriptors = mat['descriptors']
            # hist = np.array(bag_of_words_histogram(kmeansminbatch, descriptors, vocabSize))
            histograms = np.vstack((histograms, bag_of_words_histogram(kmeansminbatch, descriptors, vocabSize)))
        except:
            histograms = np.vstack((histograms, np.zeros((1, vocabSize))))

    np.save(hfilename, histograms)

# print(histograms.shape)
scores = []
print(histograms.shape)
print(len(sample_images))
for i in random.sample(range(0,len(sample_images)), 3):
    print("***************")
    scores = []
    fname = siftdir + fnames[sample_images[i]]
    print(fname)
    histA = histograms[i]
    for j in sample_images:
        if j == sample_images[i]:
            scores.append((float('-inf'), j))
        else:
            score = normalizedScalarProduct(histA, histograms[j], vocabSize)
            scores.append( (score, j) )
    best_images = []
    scores.sort(key=lambda x: -x[0])
    for j in range(M):
        best_images.append(scores[j][1])
        print(scores[j][0], scores[j][1])
    r = 2
    c = 3
    f, axarr = plt.subplots(r,c)
    curr = 0
    for j in range(r):
        for k in range(c):
            if (curr != 5):
                imname = framesdir + fnames[best_images[curr]][:-4]
                print(imname)
                im = misc.imread(imname)
                axarr[j,k].imshow(im)
                curr+=1
            else:
                imname = framesdir + fnames[sample_images[i]][:-4]
                im = misc.imread(imname)
                axarr[j,k].imshow(im)

    plt.show()
