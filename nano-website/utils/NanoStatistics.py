import streamlit as st
import random
import numpy as np
from scipy.spatial import distance

import plotly.express as px
import plotly.figure_factory as ff

#
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

#
@st.cache_data(show_spinner = False, max_entries = 5)
def uniformity(BLOBs, sizeImage, sizeBlock):
    heightBlocks = int(np.ceil(sizeImage[1] / sizeBlock))
    widthBlocks = int(np.ceil(sizeImage[0] / sizeBlock))
    
    counter = np.zeros((heightBlocks, widthBlocks), dtype = int)        
    for blob in BLOBs:
        y, x, r = blob
        i = np.ceil(y / sizeBlock) - 1
        j = np.ceil(x / sizeBlock) - 1
        counter[int(i), int(j)] += 1

    return counter

#
@st.cache_data(show_spinner = False, max_entries = 5)
def euclideanDistance(c_blobs):
    points = c_blobs[:, 0:2]
    fullEuclideanDist = distance.cdist(points, points, 'euclidean')

    nblobs = np.shape(c_blobs)[0]
    minEuclideanDist = np.min(fullEuclideanDist + np.eye(nblobs, nblobs) * 10 **6, axis = 0)

    return fullEuclideanDist, minEuclideanDist

#
@st.cache_data(show_spinner = False, max_entries = 5)
def averageDensityInNeighborhood(c_thresholds, c_fullDist):
    distanceLess = np.zeros(len(c_thresholds))

    for i, threshold in enumerate(c_thresholds):
        distanceLess[i] = (np.less(c_fullDist, threshold).sum() - len(c_fullDist)) / (np.pi * threshold**2)

    return distanceLess

#
@st.cache_data(show_spinner = False, max_entries = 5)
def calculateParametersNP(diameters, density, imageSize, scale):
    volume = (np.pi * diameters**3) / 6
    area =  np.sum((np.pi * diameters**2) / 4)
    mass = np.sum(volume * density)

    imageArea = np.prod(imageSize)
    if scale is not None:
        imageArea = imageArea * (scale**2)

    normArea = area/imageArea*100
    normMass = mass/imageArea

    return {
        "volume": np.sum(volume),
        "area": area,
        "mass": mass,
        "normArea": normArea,
        "normMass": normMass,
        "imageArea": imageArea
    }

### main
if __name__ == "__main__":

    BLOB = randon_BLOBS(2500)
    
    fullDist, minDist = euclideanDistance(BLOB)
    
    x = np.arange(5, 100, 1)

    temp_2 = averageDensityInNeighborhood(x, fullDist)

    fig = px.bar(x = x, y = temp_2)
    fig.show()
