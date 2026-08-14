import streamlit as st
import random
import numpy as np
import scipy

import plotly.express as px

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
def uniformity(x, y, blockShape, oneBlockSize):
    counter = np.zeros(blockShape, dtype = int)        
    for _x, _y in zip(x, y):
        counter[int(_y // oneBlockSize), int(_x // oneBlockSize)] += 1

    return counter


#
@st.cache_data(show_spinner = False, max_entries = 5)
def euclideanDistance(x, y):
    points = np.column_stack((x, y))

    fullEuclideanDist = scipy.spatial.distance.cdist(points, points, 'euclidean')

    size = len(x)
    minEuclideanDist = np.min(fullEuclideanDist + np.eye(size) * 10**6, axis = 0)

    return fullEuclideanDist, minEuclideanDist


#
@st.cache_data(show_spinner = False, max_entries = 5)
def localAreaFraction(c_thresholds, c_fullDist, particlesDiameter):

    particleAreas = np.pi * particlesDiameter**2 / 4
    areaFraction = np.zeros(len(c_thresholds))

    for i, threshold in enumerate(c_thresholds):
        mask = np.less(c_fullDist, threshold)
        localAreas = np.dot(mask, particleAreas)
        areaFraction[i] = np.mean(localAreas) / (np.pi * threshold**2)

    return areaFraction


#
@st.cache_data(show_spinner = False, max_entries = 5)
def averageNeighborhoods(c_thresholds, c_fullDist):
    distanceLess = np.zeros(len(c_thresholds))

    for i, threshold in enumerate(c_thresholds):
        distanceLess[i] = np.less(c_fullDist, threshold).sum() / len(c_fullDist) - 1 
        #  subtract '1' to exclude particles on diagonal

    return distanceLess


#
@st.cache_data(show_spinner = False, max_entries = 5)
def averageDensityInNeighborhood(c_thresholds, c_fullDist):
    distanceLess = np.zeros(len(c_thresholds))

    for i, threshold in enumerate(c_thresholds):
        distanceLess[i] = (np.less(c_fullDist, threshold).sum() - len(c_fullDist)) / (np.pi * threshold**2)
        #  subtract 'len(c_fullDist)' to exclude particles on diagonal

    return distanceLess / len(c_fullDist)


@st.cache_data(show_spinner = False, max_entries = 5)
def aggregateStatistics(
    statisticDiameters,
    minDist,
    materialDensity = None,
    imageArea = None,
    scaleUnit = "px",
):
    meanDiameter = np.mean(statisticDiameters)
    meanNearest = np.mean(minDist)

    if len(minDist) < 3:
        mostProbableNearest = meanNearest
    elif np.std(minDist) == 0:
        mostProbableNearest = minDist[0]
    else:
        kde = scipy.stats.gaussian_kde(minDist)
        kdeX = np.linspace(minDist.min(), minDist.max(), 1000)
        mostProbableNearest = kdeX[np.argmax(kde(kdeX))]

    threshold = round(meanDiameter)

    result = {        
        "Scale unit": scaleUnit,
        "Number of particles": len(statisticDiameters),
        "Mean particle diameter": meanDiameter,
        "Mean distance to neighbour": meanNearest,
        "Most probable distance to neighbour": mostProbableNearest,
        "Distance threshold": threshold,
        "Fraction below distance threshold": np.mean(minDist < threshold),

        # физические величины по умолчанию        
        "Particle surface density, mg/m²": None,
        "Clark-Evans index (R)": None,
    }

    if (
        scaleUnit != "px"
        and materialDensity is not None
        and imageArea is not None
    ):
        result["Particle surface density, mg/m²"] = (
            np.sum((np.pi / 6 * statisticDiameters**3) * materialDensity * 1e12)
            / imageArea
        )

        result["Clark-Evans index (R)"] = (
            2 * meanNearest
            * np.sqrt(len(statisticDiameters) / imageArea)
        )

    return result


### main
if __name__ == "__main__":

    BLOB = randon_BLOBS(2500)
    
    fullDist, minDist = euclideanDistance(BLOB)
    
    x = np.arange(5, 100, 1)

    temp_2 = averageDensityInNeighborhood(x, fullDist)

    fig = px.bar(x = x, y = temp_2)
    fig.show()
