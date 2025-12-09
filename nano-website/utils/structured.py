import math
import numpy as np
from scipy.spatial.distance import squareform
from dataclasses import dataclass

import streamlit as st


@dataclass
class Parameters:
    DENSITY_NEIGHBOUR_COUNT:    int     = 3
    DENSITY_WEIGHT:             float   = 1.5
    PCA_NEIGHBOUR_COUNT:        int     = 8
    THR_QUALITY:                float   = 0.85
    LINE_LENGTH:                int     = 7
    WEIGHT_METRIC_THR:          float   = 0.03
    WEIGHT_COAXISL:             float   = 1.75
    COAXIS_PERIOD:              int     = 6

    
@st.cache_data(show_spinner = False)
def euclidDistances(points2D): # TO DO use matrix operations
    diff = points2D[:, np.newaxis, :] - points2D[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis = 2)
    np.fill_diagonal(distances, np.inf)
    return distances

@st.cache_data(show_spinner = False)
def densityParticles(distance, settings):
    vectorDist = squareform(distance, force = 'tovector', checks = False)
    numberSmallestDistances = settings.DENSITY_NEIGHBOUR_COUNT * distance.shape[0]
    smallestDistances = np.partition(vectorDist, numberSmallestDistances)
    density = np.sum(smallestDistances[:numberSmallestDistances:]) / numberSmallestDistances
    return density * settings.DENSITY_WEIGHT


class PrevailingDirections:
    def __init__(self, point2d, settings):

        def findNearest(_pointIndex, _metric, _unusingIndexFlag, _nearestThreshold = np.inf):
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

        countParticles = point2d.shape[0]
        self.slopeCoef = np.zeros(countParticles)
        self.quality = np.zeros(countParticles)
        
        distance = euclidDistances(point2d)

        density = densityParticles(distance, settings)
    
        # create local group particles
        for i in range(countParticles):
            unusingIndexFlag = np.ones(countParticles, dtype = 'bool')
            unusingIndexFlag[i] = False
            neighbourPointsIndexs = np.array([i])

            for _ in range(settings.PCA_NEIGHBOUR_COUNT - 1):
                nearestIndex = findNearest(neighbourPointsIndexs, distance, unusingIndexFlag, density)
                if (nearestIndex != -1):
                    neighbourPointsIndexs = np.insert(neighbourPointsIndexs, 0, nearestIndex)
                    unusingIndexFlag[nearestIndex] = False
                else:
                    break
            
            # calc prevailing directions
            if len(neighbourPointsIndexs) < 3: # 3 - minimal count particles in group
                self.slopeCoef[i], self.quality[i] = (0, 0)
            else:            
                self.slopeCoef[i], self.quality[i] = trendLine(point2d[neighbourPointsIndexs, :2].T)

    def features(self, settings):  
        angle = np.arctan(self.slopeCoef) * 180 / np.pi
        indexBestQuality = self.quality >= settings.THR_QUALITY

        # 1 fraction of reliable orientations  
        feature1 = np.sum(self.quality[indexBestQuality]) / np.sum(self.quality)

        # 2 general consistency of orientations
        value, _ = np.histogram(angle, bins = range(-90, 92, 2), weights = self.quality)
        normValue = value / np.sum(value)
        H = -np.nansum(normValue * np.log2(normValue))
        feature2 = H / np.log2(len(normValue))

        # 3 consistency of best orientations
        value, _ = np.histogram(angle[indexBestQuality], bins = range(-90, 92, 2), weights = self.quality[indexBestQuality])
        normValue = value / np.sum(value)
        H = -np.nansum(normValue * np.log2(normValue))
        feature3 = H / np.log2(len(normValue))

        return feature1, feature2, feature3


### 2. shortest unclosed path



### 3. minimum spanning forest


if __name__ == "__main__":    
    import io, csv
    from types import SimpleNamespace

    tempFile = fr"C:\Cloud\42-S1-no_area-100k-ordered,Exp"

    with open(tempFile, 'r') as _file:
        reader = csv.reader(_file, delimiter = ',')
        BLOBs = np.array(list(reader), dtype = float)
        points2D = BLOBs[:, :2]
   
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

    temp = PrevailingDirections(points2D, settings)

    print(temp.features(settings))