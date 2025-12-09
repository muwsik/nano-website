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

    
def euclidDistances(points2D): # TO DO use matrix operations
    diff = points2D[:, np.newaxis, :] - points2D[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis = 2)
    np.fill_diagonal(distances, np.inf)
    return distances


def densityParticles(distance, settings):
    vectorDist = squareform(distance, force = 'tovector', checks = False)
    numberSmallestDistances = settings.DENSITY_NEIGHBOUR_COUNT * distance.shape[0]
    smallestDistances = np.partition(vectorDist, numberSmallestDistances)
    density = np.sum(smallestDistances[:numberSmallestDistances:]) / numberSmallestDistances
    return density * settings.DENSITY_WEIGHT


## 1
def calcPrevailingDirections(points2D, distance, density, settings):

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

    countParticles = points2D.shape[0]
    slopeCoef = np.zeros(countParticles)
    quality = np.zeros(countParticles)
            
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
            slopeCoef[i], quality[i] = (0, 0)
        else:            
            slopeCoef[i], quality[i] = trendLine(points2D[neighbourPointsIndexs, :2].T)

    return slopeCoef, quality

def featuresPrevailingDirections(slopeCoef, quality, settings):  
        angle = np.arctan(slopeCoef) * 180 / np.pi
        indexBestQuality = quality >= settings.THR_QUALITY

        # 1 fraction of reliable orientations  
        feature1 = np.sum(quality[indexBestQuality]) / np.sum(quality)

        # 2 general consistency of orientations
        value, _ = np.histogram(angle, bins = range(-90, 92, 2), weights = quality)
        normValue = value / np.sum(value)
        H = -np.nansum(normValue * np.log2(normValue))
        feature2 = H / np.log2(len(normValue))

        # 3 consistency of best orientations
        value, _ = np.histogram(angle[indexBestQuality], bins = range(-90, 92, 2), weights = quality[indexBestQuality])
        normValue = value / np.sum(value)
        H = -np.nansum(normValue * np.log2(normValue))
        feature3 = H / np.log2(len(normValue))

        return feature1, feature2, feature3
    
def prevailingDirectionsDistances(euclidDist, slopeCoef, quality, settings):
    error = 1 - quality
    angles = np.arctan(slopeCoef)
    C = settings.WEIGHT_METRIC_THR

    # sin(|angle_i - angle_j|)
    sin_diff = np.sin(np.abs(angles[:, None] - angles[None, :]))

    # (error_i + error_j)/2
    avg_error = (error[:, None] + error[None, :]) / 2

    return C * euclidDist + 2 * (1 - C) * np.maximum(sin_diff, avg_error)

## 2
class Line:

    class Point:
        def __init__(self, _index, _flag = True, _value = np.inf):
            self.index = _index
            self.flag = _flag
            self.value = _value

    # constructor
    def __init__(self, _startIndex = -1, _metricThreshold = -1.0, WEIGHT_COAXIS = 1.5):
        self.line = []              # curve points
        self.lineI = [_startIndex]  # point indices
        self.tail  = Line.Point(_startIndex)        
        self.head  = Line.Point(_startIndex)      
        self.start = Line.Point(_startIndex)
        self.threshold = _metricThreshold
        self.WEIGHT_COAXIS = WEIGHT_COAXIS
        
    # operator []
    def __getitem__(self, index):
        return np.array(self.line)[index]

    # add point in line nearest on end or start
    def AddPoint(self, _metric, _unusingIndexFlag, _globalIndexs, _BLOBs):
        
        def coaxis3P(_points) -> float:
            AB = math.dist(_points[ 0 ], _points[ 1 ])
            BC = math.dist(_points[ 1 ], _points[ 2 ])
            AC = math.dist(_points[ 0 ], _points[ 2 ])
            if ((AB == 0.0) or (BC == 0.0)):
                return 0.0
            else:
                return  (1 - (AB**2 + BC**2 - AC**2)/(2 * AB * BC)) / 2

        # auxiliary calculations 
        def FindNearest(side:bool):
            if side:
                i1, i2 =  0,  1
            else:
                i1, i2 = -1, -2

            tempRow = _metric[self.lineI[i1],  _unusingIndexFlag]
            if (self.Length() < 2):
                min = np.min(tempRow)
                currentGlobalIndex = (_globalIndexs[_unusingIndexFlag])[np.argmin(tempRow)]
            else:
                tempFlags = tempRow <= self.threshold
                tempGlobalIndexes = (_globalIndexs[_unusingIndexFlag])[tempFlags]
                if (len(tempGlobalIndexes)):
                    points = _BLOBs[tempGlobalIndexes, :-1]
                    corrMetric = _metric[self.lineI[i1],  tempGlobalIndexes]
                    for i, point in enumerate(points):
                        probableLine = [point, _BLOBs[self.lineI[i1], :-1], _BLOBs[self.lineI[i2], :-1]]
                        tempConcentricity = coaxis3P(probableLine)
                        corrMetric[i] += self.WEIGHT_COAXIS * (1 - tempConcentricity**2)
                    min = np.min(corrMetric)
                    currentGlobalIndex = tempGlobalIndexes[np.argmin(corrMetric)]
                else:
                    return np.inf, -1
            
            if (min > self.threshold):
                return np.inf, -1

            return min, currentGlobalIndex


        if (not any(_unusingIndexFlag)):
            return False
        
        if self.head.flag:
            self.head.value, self.head.index = FindNearest(True) 
            
        if self.tail.flag:
            self.tail.value, self.tail.index = FindNearest(False) 

        if (self.head.index != -1) and (self.head.value <= self.tail.value):
            self.lineI.insert(0, self.head.index)
            _unusingIndexFlag[self.head.index] = False

            if (self.head.index != self.tail.index):
                self.tail.flag = False
            else:
                self.tail.flag = True
            
            self.head.flag = True
            return True

        if (self.tail.index != -1) and (self.tail.value < self.head.value):            
            self.lineI.append(self.tail.index)
            _unusingIndexFlag[self.tail.index] = False

            if (self.head.index != self.tail.index):
                self.head.flag = False
            else:
                self.head.flag = True
            
            
            self.tail.flag = True
            return True

        return False

    # length of line (in points)
    def Length(self):
        return len(self.lineI)
    
    # on index points to coordinate points
    def GetLine(self, _BLOBs):
        if (self.line == []):
            self.line = _BLOBs[self.lineI, :-1]
        return self.line

    # calculate concentricity
    def Coaxis(self, _period:int) -> float: 
        steps:int = _period - 1
        if self.Length() <= steps:
            return None
               
        coaxis:float = 0
        brokeLineLength:float = 0
        for i in range(self.Length() - steps):
            if (i == 0):  
                for j in range(steps):
                    brokeLineLength += math.dist(self.line[i+j], self.line[i+j+1])
            else:
                brokeLineLength -= math.dist(self.line[i], self.line[i-1])                
                brokeLineLength += math.dist(self.line[i+steps-1], self.line[i+steps])
            
            terminalDist = math.dist(self.line[i], self.line[i+steps])   
            coaxis +=  terminalDist / brokeLineLength
        
        return coaxis / (self.Length() - steps)

def calcLineSUP(points2D, distancePD, quality, density, settings):      
    countParticles = points2D.shape[0]

    metricThresholdPD = settings.WEIGHT_METRIC_THR * density + 1

    # True (1) - not used now; False (0) - already used
    unusingIndexFlag = np.ones(countParticles, dtype = 'bool')
    globalIndexs = np.array(range(countParticles))
    lines = []

    while(any(unusingIndexFlag)):
        if (np.max(quality[unusingIndexFlag]) < settings.THR_QUALITY):
            break;

        startIndex = np.argmax(quality[unusingIndexFlag])
        startGlobalIndexs = (globalIndexs[unusingIndexFlag])[startIndex]
        unusingIndexFlag[startGlobalIndexs] = False
        tempLine = Line(startGlobalIndexs, metricThresholdPD, settings.WEIGHT_COAXIS)
  
        while (tempLine.AddPoint(distancePD, unusingIndexFlag, globalIndexs, BLOBs)):
            pass
    
        if (tempLine.Length() >= settings.LINE_LENGTH):
            tempLine.GetLine(BLOBs)
            lines.append(tempLine)
            
    return lines

def featuresLineSUP(lines, countParticles, settings):
    if len(lines) == 0:
        return 0, 0, 0, 0
    
    sumTerminalCoaxis = 0
    sumLocalCoaxis = 0
    sumLength = 0
    for tempLine in lines:
        sumTerminalCoaxis += tempLine.Coaxis(tempLine.Length()) * tempLine.Length()
        sumLocalCoaxis += tempLine.Coaxis(settings.COAXIS_PERIOD) * tempLine.Length()        
        sumLength += tempLine.Length()   
    
    # 1 Number of lines constructed of the SUP
    feature1 = len(lines)

    # 2 Rectilinearity of the SUP-lines
    feature2 = sumTerminalCoaxis / sumLength

    # 3 Smoothness of the SUP-lines
    feature3 = sumLocalCoaxis / sumLength

    # 4 The fraction of connected nanoparticles of the SUP-lines
    feature4 = sumLength / countParticles

    return feature1, feature2, feature3, feature4


# for debagging
if __name__ == "__main__":    
    import io, csv
    from types import SimpleNamespace

    tempFile = fr"D:\Projects\nano\42-S1-no_area-100k-ordered,Exp"

    with open(tempFile, 'r') as _file:
        reader = csv.reader(_file, delimiter = ',')
        BLOBs = np.array(list(reader), dtype = float)
        points2D = BLOBs[:, :2]
   
    settings = SimpleNamespace(
        DENSITY_NEIGHBOUR_COUNT = 3,
        DENSITY_WEIGHT = 1.5,
        PCA_NEIGHBOUR_COUNT = 8,
        THR_QUALITY = 0.85,
        LINE_LENGTH = 7,
        WEIGHT_METRIC_THR= 0.03,
        WEIGHT_COAXIS = 1.75,
        COAXIS_PERIOD = 6,
    )

    distE = euclidDistances(points2D)

    density = densityParticles(distE, settings)

    PD = calcPrevailingDirections(points2D, distE, density, settings)

    distPD = prevailingDirectionsDistances(distE, PD[0], PD[1], settings)

    lines = calcLineSUP(points2D, distPD, PD[1], density, settings)
    
    print(featuresPrevailingDirections(PD[0], PD[1], settings))
    print(featuresLineSUP(lines, points2D.shape[0], settings))