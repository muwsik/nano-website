import math
import ctypes
import numpy as np
from scipy.spatial.distance import squareform


def EuclideanDistances(_points): # TO DO use matrix operations
    pointsCount = np.shape(_points)[0]
    distances = np.zeros((pointsCount, pointsCount)) + np.inf
    for i in range(pointsCount):
            for j in range(i+1, pointsCount):
                distances[i, j] = math.dist( [_points[i, 0], _points[i, 1]] ,
                                             [_points[j, 0], _points[j, 1]] )
                distances[j, i] = distances[i, j]
    return distances

def FindNearest(_pointIndex, _metric, _unusingIndexFlag, _nearestThreshold = np.inf):
    tempMetric = _metric[_pointIndex[:, np.newaxis], _unusingIndexFlag]
    if (np.min(tempMetric) > _nearestThreshold):
        return -1
    else:
        nearestPointIndex = np.argmin(tempMetric) % tempMetric.shape[1]

        countUsingPoints = 0
        countUnusingPoints = -1
        for flag in _unusingIndexFlag:
            if flag:
                countUnusingPoints += 1
            else:
                countUsingPoints += 1

            if (countUnusingPoints == nearestPointIndex):
                break

        return countUsingPoints + countUnusingPoints

# _points = [[y, x], ...]
def trendLine(_points):
    covariance = np.cov(_points)
    eigvals, eigvec = np.linalg.eig(covariance)

    if (eigvals[0] > eigvals[1]):
        index = 0
    else:
        index = 1
    
    thetta = 0
    if (np.abs(eigvec[index,1]) > 10**-6):
        thetta += math.atan(eigvec[index,0] / eigvec[index,1]) * 180 / np.pi

    if thetta < 0:
        thetta += 90
    else:
        thetta -= 90

    quality = (1 - np.min(eigvals) / np.max(eigvals))**2
    quality = round(quality, 5)

    return math.tan(-thetta * np.pi / 180), quality


### 1. prevailing directions 
def calculateFeaturesPD(BLOBs, distE, settings):
    countParticles = BLOBs.shape[0]
    slopeCoef = np.zeros(countParticles)
    quality = np.zeros(countParticles)
    
    # 1.1 calc density of nanpoarticles
    vectorDist = squareform(distE, force = 'tovector', checks = False)
    numberSmallestDistances = settings.DENSITY_NEIGHBOUR_COUNT * countParticles
    smallestDistances = np.partition(vectorDist, numberSmallestDistances)
    density = np.sum(smallestDistances[:numberSmallestDistances:]) / numberSmallestDistances
    weightDensity = density * settings.DENSITY_WEIGHT

    # 1.2 create local group particles
    for i in range(countParticles):
        unusingIndexFlag = np.ones(countParticles, dtype = 'bool')
        unusingIndexFlag[i] = False
        neighbourPointsIndexs = np.array([i])

        for _ in range(settings.PCA_NEIGHBOUR_COUNT - 1):
            nearestIndex = FindNearest(neighbourPointsIndexs, distE, unusingIndexFlag, weightDensity)
            if (nearestIndex != -1):
                neighbourPointsIndexs = np.insert(neighbourPointsIndexs, 0, nearestIndex)
                unusingIndexFlag[nearestIndex] = False
            else:
                break
            
        # 1.3 calc prevailing directions
        if len(neighbourPointsIndexs) < 3: # 3 - minimal count particles in group
            slopeCoef[i], quality[i] = (0, 0)
        else:            
            slopeCoef[i], quality[i] = trendLine(BLOBs[neighbourPointsIndexs, :2].T)

    # 1.4 features PD
    angle = np.arctan(slopeCoef) * 180 / np.pi
    indexBestQuality = quality >= settings.THR_QUALITY

    # 1.4.1 fraction of reliable orientations  
    feature1 = np.sum(quality[indexBestQuality]) / np.sum(quality)

    # 1.4.2 general consistency of orientations
    value, _ = np.histogram(angle, bins = range(-90, 92, 2), weights = quality)
    normValue = value / np.sum(value)
    H = -np.nansum(normValue * np.log2(normValue))
    feature2 = H / np.log2(len(normValue))

    # 1.4.3 general consistency of orientations
    value, _ = np.histogram(angle[indexBestQuality], bins = range(-90, 92, 2), weights = quality[indexBestQuality])
    normValue = value / np.sum(value)
    H = -np.nansum(normValue * np.log2(normValue))
    feature3 = H / np.log2(len(normValue))

    return (feature1, feature2, feature3), angle, slopeCoef, quality


### 2. shortest unclosed path



### 3. minimum spanning forest


if __name__ == "__main__":    
    import io, csv
    from types import SimpleNamespace

    tempFile = fr"D:\Projects\nano\42-S1-no_area-100k-ordered,Exp"

    with open(tempFile, 'r') as _file:
        reader = csv.reader(_file, delimiter = ',')
        BLOBs = np.array(list(reader), dtype = float)
        points2D = BLOBs[:, :2]

    distE = EuclideanDistances(points2D)
                
    settings = SimpleNamespace(
                    DENSITY_NEIGHBOUR_COUNT = 3,
                    DENSITY_WEIGHT = 1.5,
                    PCA_NEIGHBOUR_COUNT = 8,
                    THR_QUALITY = 0.85,
                    #LINE_LENGTH = 7,
                    #WEIGHT_COAXIS = 1.75,
                    #THR_LINE_START = 0.85,
                    #COAXIS_PERIOD = 6,
                    #METRIC_THRESHOLD = 0.03 * _density + 1
                )

    temp = calculateFeaturesPD(points2D, distE, settings)

    print(temp[0])