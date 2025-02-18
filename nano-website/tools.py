import streamlit as st
import MicFunctions_v2 as mf
import random
import numpy as np
from scipy.spatial import distance

@st.cache_data(show_spinner = False)
def CACHE_FindThresPrep(img, nbr, thrPrepCoef):
    return mf.FindThresPrep(img, nbr, thrPrepCoef)

@st.cache_data(show_spinner = "Nanoparticle detection on process...")
def CACHE_ExponentialApproximationMask(img, c1s, points, maskMain, wsize, thresCoefOld, nproc):
    return mf.ExponentialApproximationMask(img, c1s, points, maskMain, wsize, thresCoefOld, nproc)

def randon_BLOBS(count = 250, type = 'uniform', x_max = 1280, y_max = 890):
    fake_BLOBS = np.zeros((count, 3))

    for i in range(len(fake_BLOBS)):
        fake_BLOBS[i, 0] = random.randint(0, y_max)
        fake_BLOBS[i, 1] = random.randint(0, x_max)

    if type == 'uniform':        
        for i in range(len(fake_BLOBS)):       
            fake_BLOBS[i, 2] = random.uniform(0, 7)
    elif type == 'norm':
        fake_BLOBS[:, 2] = np.random.normal(3.5, 2.5, size = count)
    else:
        raise Exception('!')

    return fake_BLOBS


def uniformity(BLOBs, sizeImage, sizeBlock):
    heightBlocks = int(np.ceil(sizeImage[0] / sizeBlock))
    widthBlocks = int(np.ceil(sizeImage[1] / sizeBlock))
    
    counter = np.zeros((heightBlocks, widthBlocks), dtype = int)        
    for blob in BLOBs:
        y, x, r = blob
        i = np.ceil(y / sizeBlock) - 1
        j = np.ceil(x / sizeBlock) - 1
        counter[int(i), int(j)] += 1

    return counter

def euclideanDistance(_blobs):
    points = _blobs[:, 0:2]
    fullEuclideanDist = distance.cdist(points, points, 'euclidean')

    nblobs = np.shape(_blobs)[0]
    minEuclideanDist = np.min(fullEuclideanDist + np.eye(nblobs, nblobs) * 10 **6, axis = 0)

    return fullEuclideanDist, minEuclideanDist

def thresholdDistance(_thresholds, _fullDist):
    distanceLess = np.zeros(len(_thresholds))

    for i, threshold in enumerate(_thresholds):
        distanceLess[i] = np.less(_fullDist, threshold).sum() / (len(_fullDist) * np.pi * threshold ** 2)

    return distanceLess
