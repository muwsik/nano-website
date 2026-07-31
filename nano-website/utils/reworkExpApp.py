import streamlit as st

from PIL import Image, ImageDraw
import warnings
import numpy as np
import scipy, skimage
    
import utils.ExponentialApproximation as ExpApp
import utils.ExponentialApproximation2 as ExpApp2

class Particle:
    def __init__(self, centerCoords, diameter, c0 = None, approxError = None):
        self.x = centerCoords[0]
        self.y = centerCoords[1]

        if (diameter > 0):
            self.diameter = diameter
        else:
            warnings.warn("The particle diameter is less than zero!")            
            self.diameter = 0

        self.projectionArea = 1 / 4 * np.pi * self.diameter**2 

        self.volume = 2 / 3 * self.projectionArea * self.diameter

        # detection features 
        self.c0 = c0
        self.approxError = approxError
    
    # Converts the calculated values according to the multiplier scale
    def convert(self, multiplier):
        return Particle(
            (self.x * multiplier, self.y * multiplier),
            self.diameter * multiplier,
            self.c0,
            self.approxError
        )


    def toList(self):
        return [
            self.x,
            self.y,
            self.diameter,
            self.c0,
            self.approxError,
            self.projectionArea,
            self.volume,
        ]


    def toDict(self):
        return self.__dict__.copy()


    @staticmethod
    def fromArray(BLOBs):
        if len(BLOBs) < 1:
            return np.array([])

        return np.array([
             #           x,     y,      d
            Particle((_i[1], _i[0]), _i[2]) for _i in BLOBs
        ])

    @staticmethod
    def toArray(particles):
        if len(particles) < 1:
            return None

        return np.array([
            [_i.y, _i.x, _i.diameter] for _i in particles
        ])



@st.cache_data(show_spinner = False, max_entries = 5)
def detectingParticles(image, settings):

    if isinstance(image, np.ndarray):
        pass
    elif isinstance(image, Image.Image):
        image = np.array(image, dtype = 'uint8')
    else:
        raise ValueError("!")

    if (settings[0] == 2):
        image = 255 - image # inversion for TEM-image


    if settings[2] == 0:
        # параметры в пикселях
        params = {
            # размер окна медианного фильтра
            "sz_med" : 3,
            # параметр функции Гаусса для сглаживания
            "sigma_gauss": None,
            # размер диска Top-Hat
            "sz_th":  4,
            # порог яркости для отбрасывания лок. максимумов
            "thr_br": float(settings[1]),   
            # минимальное расстояние между локальными максимумами при их поиске 
            "min_dist": 3,
            # размер окна аппроксимации
            "wsize": 7,     
            # выбор лучшей точки в окрестности лок.макс. по norm_error (1 - по с1, 2 - по с0, 3 - по norm_error) 
            "best_mode": 3, 
            # берем окошко такого размера с центром в точке локального максимума для уточнения положения наночастицы   
            "msk": 3,      
            # аппроксимирующая функция "exp" или "pol" 
            "met": 'exp',   
            # число параметров аппроксимации
            "npar": 2,
            # потенциальное количество наночастиц
            "nlocmax": 5000,
        }
    elif settings[2] == 1:
        # параметры в пикселях
        params = {
            # размер окна медианного фильтра
            "sz_med" : 3,
            # параметр функции Гаусса для сглаживания
            "sigma_gauss": 1,
            # размер диска Top-Hat
            "sz_th":  6,
            # порог яркости для отбрасывания лок. максимумов
            "thr_br": float(settings[1]),   
            # минимальное расстояние между локальными максимумами при их поиске 
            "min_dist": 3,
            # размер окна аппроксимации
            "wsize": 9,     
            # выбор лучшей точки в окрестности лок.макс. по norm_error (1 - по с1, 2 - по с0, 3 - по norm_error) 
            "best_mode": 3, 
            # берем окошко такого размера с центром в точке локального максимума для уточнения положения наночастицы   
            "msk": 3,      
            # аппроксимирующая функция "exp" или "pol" 
            "met": 'exp',   
            # число параметров аппроксимации
            "npar": 2,
            # потенциальное количество наночастиц
            "nlocmax": 1500,
        }
    elif settings[2] == 2:
        # параметры в пикселях
        params = {
            # размер окна медианного фильтра
            "sz_med" : 3,
            # параметр функции Гаусса для сглаживания
            "sigma_gauss": 1.5,
            # размер диска Top-Hat
            "sz_th":  8,
            # порог яркости для отбрасывания лок. максимумов
            "thr_br": float(settings[1]),   
            # минимальное расстояние между локальными максимумами при их поиске 
            "min_dist": 3,
            # размер окна аппроксимации
            "wsize": 7,     
            # выбор лучшей точки в окрестности лок.макс. по norm_error (1 - по с1, 2 - по с0, 3 - по norm_error) 
            "best_mode": 3, 
            # берем окошко такого размера с центром в точке локального максимума для уточнения положения наночастицы   
            "msk": 3,      
            # аппроксимирующая функция "exp" или "pol" 
            "met": 'exp',   
            # число параметров аппроксимации
            "npar": 2,
            # потенциальное количество наночастиц
            "nlocmax": 700,
        }
    else:
        raise ValueError("!")
    
    image = ExpApp.PreprocessingMedian(image, params['sz_med'])
    image = ExpApp.PreprocessingTopHat(image, params['sz_th'])
    if params["sigma_gauss"] is not None:
        image = scipy.ndimage.gaussian_filter(
            image if settings[2] == 1 else image[::2, ::2],
            sigma = params["sigma_gauss"]
        )

    nlocmax = params["nlocmax"]
    numpeaks = max(1000, nlocmax)
    lms = skimage.feature.peak_local_max(image,
        min_distance = params["min_dist"],
        threshold_abs = params["thr_br"],
        threshold_rel = None,
        footprint = None,
        labels = None,
        num_peaks = numpeaks
    )
    lmblobs = lms[:nlocmax]
                                  
    blobs_appr = np.array(ExpApp2.ApproximationMain(image, lmblobs, params, 3, True))
                
    if len(blobs_appr) < 1:
        return np.array([])

    if settings[2] == 2:
        blobs_appr[:,:3] *= 2
    elif settings[2] == 1 or settings[2] == 0:
        pass
    else:
        raise ValueError("!")

    result = []
    for temp in blobs_appr:
        new = Particle((temp[1], temp[0]), temp[2]*2, temp[3], temp[5])
        result.append(new)

    return np.array(result)


@st.cache_data(show_spinner = False, max_entries = 5)
def paintParticles(image, particles, color):
    draw = ImageDraw.Draw(image)
    for _temp in particles:                
        y = _temp.y; x = _temp.x; r = _temp.diameter/2        
        draw.ellipse((x-r, y-r, x+r, y+r), outline = color)

    return image


@st.cache_data(show_spinner = False, max_entries = 5)
def filtrationParticles(particles, 
    c0 = (None, None), 
    diameter = (None, None), 
    approxError = (None, None)
):
    def inRange(value, limits):
        min_val, max_val = limits
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True

    filteredParticles = []
    for _particle in particles:       

        if not inRange(_particle.diameter, diameter):
            continue

        if not inRange(_particle.c0, c0):
            continue

        if not inRange(_particle.approxError, approxError):
            continue
    
        filteredParticles.append(_particle)

    return np.array(filteredParticles)