import numpy as np
import glob
import scipy.io
from scipy import misc
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
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
import warnings

def k_mean_distance(data, cx, cy, i_centroid, cluster_labels):
        distances = [np.sqrt((x-cx)**2+(y-cy)**2) for (x, y) in data[cluster_labels == i_centroid]]
        return distances

warnings.filterwarnings("ignore", category=DeprecationWarning)


framesdir = 'frames/'
siftdir = 'sift/'
vocabSize = 1500
num_trainImages = 3000
num_samples = 25

filename = 'finalized_model_kmeans' + str(vocabSize) + '.sav'
tfilename = 'trainging' + str(vocabSize)
kfilename = 'inversetable' + str(vocabSize)
kmeansminbatch = None
sets = None
fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]
try:
    kmeansminbatch = pickle.load(open(filename, 'rb'))
    sample_images = pickle.load(open(tfilename, 'rb'))
except:
    kmeansminbatch = MiniBatchKMeans(n_clusters = vocabSize, batch_size=5000)
    sample_images = list(range(0, num_trainImages))
    for i in sample_images:
        fname = siftdir + fnames[i]
        try:
            mat = scipy.io.loadmat(fname)
            descriptors = mat['descriptors']
            kmeansminbatch = kmeansminbatch.partial_fit(descriptors)
        except:
            continue
    pickle.dump(kmeansminbatch, open(filename, 'wb'))
    pickle.dump(sample_images, open(tfilename, 'wb'))


print("trained and saved")

try:
    sets = pickle.load(open(kfilename, 'rb'))
except:
    sets = [set() for z in range(vocabSize)]
    for i in sample_images:
        fname = siftdir + fnames[i]
        try:
            mat = scipy.io.loadmat(fname)
            descriptors = mat['descriptors']
            mappings = kmeansminbatch.predict(descriptors)
            for map in mappings:
                sets[map].add(i)
        except:
            continue
    pickle.dump(sets, open(kfilename, 'wb'))
sets.sort(key=lambda x: len(x))
print("inverse created")
number_of_elements = 3
chosenwords = [0, 1]
for i in range(0, vocabSize):
    if len(sets[i]) >= 25:
        chosenwords[0] = i
        chosenwords[1] = i + 1
        break
samples = [[],[]]
centroids = kmeansminbatch.cluster_centers_
for i in range(2):
    dataset = []
    rand = 0
    for img in sets[chosenwords[i]]:
        fname = siftdir + fnames[img]
        mat = scipy.io.loadmat(fname)
        descriptors = mat['descriptors']
        mappings = kmeansminbatch.predict(descriptors)
        scores = dist2(mat['descriptors'][:,:], centroids)
        for j in range(len(mappings)):
            if (mappings[j] == chosenwords[i]):
                a = (mat['descriptors'][j,:] , img, scores[j][mappings[j]], j)
                dataset.append(a)
        if (len(dataset) is not 0 and rand >= 5):
            dataset.sort(key=lambda x: -x[2])
            curr = 0
            while (len(samples[i]) < num_samples and curr < 5 and curr < len(dataset)):
                data = dataset[curr]
                fname = siftdir + fnames[data[1]]
                mat = scipy.io.loadmat(fname)
                imname = framesdir + fnames[data[1]][:-4]
                im = misc.imread(imname)
                img_patch = getPatchFromSIFTParameters(mat['positions'][data[3],:], mat['scales'][data[3],:], mat['orients'][data[3],:], rgb2gray(im))
                samples[i].append(img_patch)
                curr+=1
            dataset = []
            rand = 0
            if(len(samples[i]) == num_samples):
                break
        else:
            rand += 1

print("samples created")
r = 5
c = 5
f, axarr = plt.subplots(r,c)
curr = 0
for k in range(2):
    print(k)
    for i in range(r):
        for j in range(c):
            axarr[i,j].imshow(samples[k][curr], cmap='gray', aspect='auto')
            curr+=1
    plt.show()
    f, axarr = plt.subplots(r,c)
    curr = 0
