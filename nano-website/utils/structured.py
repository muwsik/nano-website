# -*- coding: cp1251 -*-
import math
import numpy as np
from scipy.spatial.distance import squareform
from dataclasses import dataclass

import matplotlib.pyplot as plt

## Settings method
@dataclass
class Parameters:
    DENSITY_NEIGHBOUR_COUNT:    int     = 3
    DENSITY_WEIGHT:             float   = 1.5
    PCA_NEIGHBOUR_COUNT:        int     = 8
    THR_QUALITY:                float   = 0.85
    MIN_LINE_SUP_LENGTH:        int     = 7
    WEIGHT_METRIC_THR:          float   = 0.03
    WEIGHT_COAXISL:             float   = 1.75
    COAXIS_PERIOD:              int     = 6
    MIN_LINE_MSF_LENGTH:        int     = 5
    MAX_DISTANCE:               float   = 20.0
    NUMBER_LONGEST_LINE:        int     = 20

## Main
class Structured:
    def __init__(self, points2D, settings):
        self._points2D = points2D.copy()
        self._settings = settings

        self._distE = None       # Euclid distances
        self._density = None     # density nanoparticles
        self._k = None           # slope coefficient a straight line (y = k * x)
        self._quality = None     # accuracy of definition slope coefficient
        self._distPD = None      # Prevailing Directions distances
        self._lineSUP = None     # lines constructed of the SUP
        #self._MSF_E = None       # MSF-lines in Euclid metric
        #self._MSF_PD = None      # MSF-lines in Prevailing Directions metric
    
    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, newSettings):
        if newSettings.DENSITY_NEIGHBOUR_COUNT != self._settings.DENSITY_NEIGHBOUR_COUNT:
            self._density = None
        
        if newSettings.PCA_NEIGHBOUR_COUNT != self._settings.PCA_NEIGHBOUR_COUNT:
            self._k = None
            self._quality = None

        if newSettings.WEIGHT_METRIC_THR != self._settings.WEIGHT_METRIC_THR:
            self._distPD = None
            self._lineSUP = None

        if (newSettings.THR_QUALITY != self._settings.THR_QUALITY) or \
        (newSettings.WEIGHT_COAXIS != self._settings.WEIGHT_COAXIS):
            self._lineSUP = None

        

    @property
    def distE(self):
        if self._distE is None:
            diff = self._points2D[:, np.newaxis, :] - self._points2D[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis = 2)
            np.fill_diagonal(distances, np.inf)
            self._distE = distances

        return self._distE

    @property
    def density(self):        
        if self._density is None:
            vectorDist = squareform(self.distE, force = 'tovector', checks = False)
            numberSmallestDistances = self._settings.DENSITY_NEIGHBOUR_COUNT * self._points2D.shape[0]
            smallestDistances = np.partition(vectorDist, numberSmallestDistances)
            self._density = np.sum(smallestDistances[:numberSmallestDistances:]) / numberSmallestDistances
        
        return self._density * self._settings.DENSITY_WEIGHT

    @property
    def prevailingDirections(self):
        if (self._k is None) or (self._quality is None):

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

            countPoints = self._points2D.shape[0]
            k = np.zeros(countPoints)
            quality = np.zeros(countPoints)
            
            # create local group particles
            for i in range(countPoints):
                unusingIndexFlag = np.ones(countPoints, dtype = 'bool')
                unusingIndexFlag[i] = False
                neighbourPointsIndexs = np.array([i])

                for _ in range(self._settings.PCA_NEIGHBOUR_COUNT - 1):
                    nearestIndex = findNearest(neighbourPointsIndexs, self.distE, unusingIndexFlag, self.density)
                    if (nearestIndex != -1):
                        neighbourPointsIndexs = np.insert(neighbourPointsIndexs, 0, nearestIndex)
                        unusingIndexFlag[nearestIndex] = False
                    else:
                        break
            
                # calc prevailing directions
                if len(neighbourPointsIndexs) < 3: # 3 - minimal count particles in group
                    k[i], quality[i] = (0, 0)
                else:            
                    k[i], quality[i] = trendLine(self._points2D[neighbourPointsIndexs, :2].T)

            self._k = k
            self._quality = quality

        return  self._k, self._quality
    
    @property
    def featuresPrevailingDirections(self):
        self.prevailingDirections

        angle = np.arctan(self._k) * 180 / np.pi
        indexBestQuality = self._quality >= self._settings.THR_QUALITY

        # 1 fraction of reliable orientations  
        feature1 = np.sum(self._quality[indexBestQuality]) / np.sum(self._quality)

        # 2 general consistency of orientations
        value2, _ = np.histogram(angle, bins = range(-90, 92, 2), weights = self._quality)
        normValue2 = value2 / np.sum(value2)
        H2 = -np.nansum(normValue2 * np.log2(normValue2 + 1e-12))
        feature2 = H2 / np.log2(len(normValue2))

        # 3 consistency of best orientations
        value3, _ = np.histogram(angle[indexBestQuality], bins = range(-90, 92, 2), weights = self._quality[indexBestQuality])
        normValue3 = value3 / np.sum(value3)
        H3 = -np.nansum(normValue3 * np.log2(normValue3 + 1e-12))
        feature3 = H3 / np.log2(len(normValue3))

        return feature1, feature2, feature3

    @property
    def distPD(self):
        if self._distPD is None:
            self.prevailingDirections

            error = 1 - self._quality
            angles = np.arctan(self._k)
            C = self._settings.WEIGHT_METRIC_THR

            # sin(|angle_i - angle_j|)
            sin_diff = np.sin(np.abs(angles[:, None] - angles[None, :]))

            # (error_i + error_j)/2
            avg_error = (error[:, None] + error[None, :]) / 2

            self._distPD = C * self.distE + (1 - C) * 2 * np.maximum(sin_diff, avg_error)

        return self._distPD

    @property
    def lineSUP(self):
        if self._lineSUP is None:

            class Line:
                class Point:
                    def __init__(self, _index, _flag = True, _value = np.inf):
                        self.index = _index
                        self.flag = _flag
                        self.value = _value

                # constructor
                def __init__(self, _startIndex = -1, _metricThreshold = -1.0, WEIGHT_COAXIS = 1.75):
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
                                points = _BLOBs[tempGlobalIndexes]
                                corrMetric = _metric[self.lineI[i1],  tempGlobalIndexes]
                                for i, point in enumerate(points):
                                    probableLine = [point, _BLOBs[self.lineI[i1]], _BLOBs[self.lineI[i2]]]
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
                        self.line = _BLOBs[self.lineI]

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
                    
            self.prevailingDirections
            countParticles = self._points2D.shape[0]
            metricThresholdPD = self._settings.WEIGHT_METRIC_THR * self.density + 1

            # True (1) - not used now; False (0) - already used
            unusingIndexFlag = np.ones(countParticles, dtype = 'bool')
            globalIndexs = np.array(range(countParticles))
            lines = []

            while(any(unusingIndexFlag)):
                if (np.max(self._quality[unusingIndexFlag]) < self._settings.THR_QUALITY):
                    break;

                startIndex = np.argmax(self._quality[unusingIndexFlag])
                startGlobalIndexs = (globalIndexs[unusingIndexFlag])[startIndex]
                unusingIndexFlag[startGlobalIndexs] = False
                tempLine = Line(startGlobalIndexs, metricThresholdPD, self._settings.WEIGHT_COAXIS)
  
                while (tempLine.AddPoint(self.distPD, unusingIndexFlag, globalIndexs, self._points2D)):
                    pass    
                
                tempLine.GetLine(self._points2D)
                lines.append(tempLine)
            
            self._lineSUP = lines

        return [
            tempLine for tempLine in self._lineSUP 
                if tempLine.Length() >= self._settings.MIN_LINE_SUP_LENGTH
        ]

    @property
    def featuresLineSUP(self):
        filtereLine = self.lineSUP

        # 1 Number of lines constructed of the SUP
        feature1 = len(filtereLine)

        if feature1 == 0:
            return 0, 0, 0, 0
    
        sumTerminalCoaxis = 0
        sumLocalCoaxis = 0
        sumLength = 0
        for tempLine in filtereLine:
            sumTerminalCoaxis += tempLine.Coaxis(tempLine.Length()) * tempLine.Length()
            sumLocalCoaxis += tempLine.Coaxis(self._settings.COAXIS_PERIOD) * tempLine.Length()        
            sumLength += tempLine.Length()   
    
        # 2 Smoothness of the SUP-lines
        feature2 = sumLocalCoaxis / sumLength
    
        # 3 Rectilinearity of the SUP-lines
        feature3 = sumTerminalCoaxis / sumLength

        # 4 The fraction of connected nanoparticles of the SUP-lines
        print(sumLength, self._points2D.shape[0])
        feature4 = sumLength / self._points2D.shape[0]

        return feature1, feature2, feature3, feature4

    
# ----------------------------------------
# GPT-MSF o_0
# ----------------------------------------

def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1
        return True

# ----------------------------------------
# 1. Краскал — построение МСТ
# ----------------------------------------

def kruskal_mst(points):
    n = len(points)
    edges = []
    
    # формируем все рёбра
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist(points[i], points[j]), i, j))
            
    edges.sort()
    dsu = DSU(n)
    
    mst = []
    for w, u, v in edges:
        if dsu.union(u, v):
            mst.append((u, v, w))
    return mst

# ----------------------------------------
# 2. Превращение МСТ в лес по порогу
# ----------------------------------------

def mst_to_forest(mst, threshold):
    return [e for e in mst if e[2] <= threshold]

# ----------------------------------------
# 3. Удаление рёбер: терминал — вершина степени > 2
# ----------------------------------------

def remove_terminal_to_highdegree_edges(forest_edges, n_points):
    degree = [0] * n_points
    
    # считаем степени
    for u, v, w in forest_edges:
        degree[u] += 1
        degree[v] += 1

    cleaned = []
    for u, v, w in forest_edges:
        u_term = degree[u] == 1
        v_term = degree[v] == 1

        u_big = degree[u] > 2
        v_big = degree[v] > 2

        forbidden = (u_term and v_big) or (v_term and u_big)

        if not forbidden:
            cleaned.append((u, v, w))

    return cleaned

# ----------------------------------------
# 4. Визуализация
# ----------------------------------------

def visualize_forest_with_long_segments(ax, points, forest_edges, segments, min_length=3, title="Длинные сегменты леса"):

    # 1) Все рёбра леса красным
    for u, v, w in forest_edges:
        x1, y1 = points[u][1], points[u][0]
        x2, y2 = points[v][1], points[v][0]
        ax.plot([x1, x2], [y1, y2], color = "red", linewidth = 1)

    # 2) Только длинные сегменты синим
    for segment in segments:
        if len(segment) > min_length:
            for i in range(len(segment) - 1):
                u, v = segment[i], segment[i+1]
                x1, y1 = points[u][1], points[u][0]
                x2, y2 = points[v][1], points[v][0]
                ax.plot([x1, x2], [y1, y2], color = "blue", linewidth = 1.5)
                
    ax.set_axis_off()
    plt.tight_layout(pad=1.75)
    plt.gca().invert_yaxis()  


from collections import defaultdict

def extract_segments(forest_edges, n_points):
    # 1. строим граф
    graph = defaultdict(list)
    degree = [0] * n_points
    for u, v, w in forest_edges:
        graph[u].append(v)
        graph[v].append(u)
        degree[u] += 1
        degree[v] += 1

    visited = [False] * n_points
    segments = []

    for start in range(n_points):
        # начинаем только с терминалов или ветвлений (degree != 2)
        if degree[start] != 2 and not visited[start]:
            for neighbor in graph[start]:
                if visited[neighbor]:
                    continue

                segment = [start, neighbor]
                visited[start] = True
                visited[neighbor] = True

                current = neighbor
                prev = start

                # идем пока degree == 2
                while degree[current] == 2:
                    next_nodes = [n for n in graph[current] if n != prev]
                    if not next_nodes:
                        break
                    nxt = next_nodes[0]
                    segment.append(nxt)
                    prev, current = current, nxt
                    visited[current] = True

                segments.append(segment)
    
    return segments

def coaxis_segment(points: np.ndarray, segment: list[int], period: int | None = None) -> float:
    """
    Вычисление коаксиальности сегмента линии.
    
    points: массив координат (N,2)
    segment: список индексов точек в сегменте
    period: длина окна для вычисления. Если None, используется длина сегмента (полный охват)
    """
    n = len(segment)
    if period is None:
        period = n
    steps = period - 1

    if n <= steps:
        return None  # сегмент слишком короткий

    coaxis = 0
    brokeLineLength = 0

    for i in range(n - steps):
        if i == 0:
            # длина "разбитой" линии для первого окна
            for j in range(steps):
                p1 = points[segment[i + j]]
                p2 = points[segment[i + j + 1]]
                brokeLineLength += math.dist(p1, p2)
        else:
            # корректируем длину, сдвигая окно
            p_old = points[segment[i - 1]]
            p_first = points[segment[i]]
            brokeLineLength -= math.dist(p_old, p_first)

            p_last_prev = points[segment[i + steps - 1]]
            p_last = points[segment[i + steps]]
            brokeLineLength += math.dist(p_last_prev, p_last)

        # расстояние между терминалами окна
        p_start = points[segment[i]]
        p_end = points[segment[i + steps]]
        terminalDist = math.dist(p_start, p_end)

        coaxis += terminalDist / brokeLineLength

    return coaxis / (n - steps)

def coaxis_all_segments_two_modes_threshold(points: np.ndarray, segments: list[list[int]], period_fixed: int, min_length: int):
    """
    Вычисление двух видов коаксиальности для сегментов, длина которых > min_length:
    - с фиксированным периодом period_fixed
    - с полным периодом (всего сегмента)

    Возвращает два списка: coaxis_fixed, coaxis_full
    """
    coaxis_fixed = []
    coaxis_full = []

    for seg in segments:
        if len(seg) >= min_length:
            value_fixed = coaxis_segment(points, seg, period_fixed)
            value_full = coaxis_segment(points, seg, None)  # None => весь сегмент                
            coaxis_fixed.append(value_fixed)
            coaxis_full.append(value_full)

    return coaxis_fixed, coaxis_full

def count_points_segments_over_threshold(segments: list[list[int]], min_length: int) -> int:
    """
    Считает количество точек в сегментах, длина которых > min_length.
    Если таких сегментов нет — считает по всем сегментам.

    segments: список сегментов (каждый сегмент — список индексов точек)
    min_length: порог длины сегмента
    """
    # выбираем сегменты, удовлетворяющие порогу
    filtered = [seg for seg in segments if len(seg) > min_length]

    # если нет сегментов выше порога — использовать все сегменты
    if not filtered:
        filtered = segments

    # считаем количество точек
    total_points = sum(len(seg) for seg in filtered)

    return total_points

def average_length_top_segments(segments: list[list[int]], N: int) -> float | None:
    """
    Считает среднюю длину (в точках) в N самых длинных сегментах.
    Если сегментов меньше N — берёт все.

    segments: список сегментов (каждый сегмент — список индексов точек)
    N: сколько самых длинных сегментов учитывать

    Возвращает float или None если сегментов нет.
    """
    if not segments:
        return 0

    # сортируем по убыванию длины
    segments_sorted = sorted(segments, key=len, reverse=True)

    # выбираем N самых длинных (или меньше, если сегментов мало)
    top_segments = segments_sorted[:N]

    # считаем длины
    lengths = [len(seg) for seg in top_segments]

    if not lengths:
        return 0

    return (sum(lengths) / len(lengths)) / len(segments_sorted[0])


def showBlobs(_blobs, _ax, color='k') -> None:
    for blob in _blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=0.5, fill=False)
        _ax.add_patch(c)   
    _ax.set_axis_off()
    plt.tight_layout(pad=1.75)


# for debagging
# if __name__ == "__main__":    
#     import io, csv
#     from types import SimpleNamespace

#     import math

    

# # ----------------------------------------
# # Пример использования
# # ----------------------------------------
#       # пример

    
#     tempFile = fr"D:\Projects\nano\42-S1-no_area-100k-ordered,Exp"
#     fileImg = fr"D:\Projects\nano\white.png"

#     with open(tempFile, 'r') as _file:
#         reader = csv.reader(_file, delimiter = ',')
#         temp_BLOBs = np.array(list(reader), dtype = float)
#         points2D = temp_BLOBs[:, :2]


#     # 1) МСТ
#     mst = kruskal_mst(points2D)

#     # 2) Лес по порогу
#     threshold = 30
#     forest = mst_to_forest(mst, threshold)

#     # 3) Удаление рёбер (терминал — вершина >2)
#     forest_clean = remove_terminal_to_highdegree_edges(forest, len(points2D))

#     # 4) Визуализация
#     segments = extract_segments(forest_clean, len(points2D))
#     visualize_forest_with_long_segments(points2D, forest_clean, segments, min_length=7)

#     min_length = 7
#     period_fixed = 5

#     coaxis_fixed, coaxis_full = coaxis_all_segments_two_modes_threshold(points2D, segments, period_fixed, min_length)

#     print("Коаксиальность с фиксированным периодом:", coaxis_fixed)
#     print("Коаксиальность с полным периодом:", coaxis_full)

    # settings = SimpleNamespace(
    #     DENSITY_NEIGHBOUR_COUNT = 3,
    #     DENSITY_WEIGHT = 1.5,
    #     PCA_NEIGHBOUR_COUNT = 8,
    #     THR_QUALITY = 0.85,
    #     LINE_LENGTH = 7,
    #     WEIGHT_METRIC_THR= 0.03,
    #     WEIGHT_COAXIS = 1.75,
    #     COAXIS_PERIOD = 6,
    # )

    # tempObj = Structured(points2D, settings)

    # print(tempObj.featuresPrevailingDirections)
    # print(tempObj.featuresLineSUP)
    
    # import matplotlib.pyplot as plt
    # import cv2

    # fig, ax = plt.subplots(1, 1, figsize=(8,10), sharex=True, sharey=True)
    # ax.imshow(cv2.imread(fileImg)[:890,:], cmap='gray')
    # for blob in points2D:
    #     y, x = blob
    #     c = plt.Circle((x, y), 1, color='k', linewidth=0.5, fill=False)
    #     ax.add_patch(c)   
    # ax.set_axis_off()
    # plt.tight_layout(pad=1.75)

    # tempSumLength = 0
    # colors = ['k', 'k', 'k']
    # for i, tempLine in enumerate(tempObj.lineSUP):
    #     tempSumLength += tempLine.Length()
    #     y, x = points2D[tempLine.start.index]
    #     ax.add_patch(plt.Circle((x, y), 2, color='r', linewidth=1, fill=False)) 

    #     plt.text(x, y, str(i), color='white',fontsize = 8)
    #     plt.plot(tempLine[:, 1], tempLine[:, 0], color=colors[i%3]) 

    # plt.show()