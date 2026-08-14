import streamlit as st

from PIL import Image, ImageDraw
import numpy as np
import scipy, skimage

from dataclasses import dataclass
    
import utils.ExponentialApproximation as ExpApp
import utils.ExponentialApproximation2 as ExpApp2


# Representation of a single detected particle
@dataclass(slots = True)
class Particle:
    x: float
    y: float
    diameter: float
    c0: float 
    approx_error: float 


# Collection of detected particles.
# Raw particle parameters are stored as NumPy arrays in pixels.
# Scale-dependent parameters are calculated once for the current
# scale and cached.
class ParticleSet:
    def __init__(self,
        blobs, # BLOBs format: [x, y, diameter, c0, approx_error]
        multiplier = None,
    ):
        self._columns = {
            "x_px": 0,
            "y_px": 1,
            "diameter_px": 2,
            "c0": 3,
            "approx_error": 4,
        }
        
        if blobs.size == 0:
            self._data = np.empty((0, 5), dtype = float)
        elif blobs.shape[1] == 3:
            self._data = np.column_stack([
                blobs,
                np.full((len(blobs), 2), -1.0),
            ])
        elif blobs.shape[1] == 5:
            self._data = blobs.astype(float, copy = False)
        else:
            raise ValueError("Expected particle data with 3 or 5 columns.")

        self._mask = np.ones(len(self._data), dtype = bool)
        self.scale_multiplier = -1.0
        self._cache = {}
        
        if multiplier is not None:
            self.applyScale(multiplier)


    def get(self, name, apply_mask = True):
        if name in self._cache:
            data = self._cache[name]
        elif name in self._columns:
            data = self._data[:, self._columns[name]]
        else:
            raise KeyError(f"Unknown particle parameter: {name}")

        if apply_mask:
            return data[self._mask]

        return data


    @property
    def count(self):
        return np.sum(self._mask)
    
    @property
    def detectedCount(self):
        return len(self._data)


    def applyScale(self, multiplier):
        if (np.isclose(multiplier, self.scale_multiplier)):
            return

        self.scale_multiplier = multiplier
        self._cache["x"] = (self._data[:, 0] * multiplier)        
        self._cache["y"] = (self._data[:, 1] * multiplier)
        self._cache["diameter"] = (self._data[:, 2] * multiplier)
        self._cache["area"] = (np.pi * self._cache["diameter"]**2 / 4)
        self._cache["volume"] = (np.pi * self._cache["diameter"]**3 / 6)


    def setfilter(self,
        c0 = (None, None),
        diameter = (None, None),
        approxError = (None, None),
    ):
        mask = np.ones(len(self._data), dtype = bool)

        if c0[0] is not None:
            mask &= self._data[:, 3] >= c0[0]
        if c0[1] is not None:
            mask &= self._data[:, 3] <= c0[1]
        if diameter[0] is not None:
            mask &= self._cache["diameter"] >= diameter[0]
        if diameter[1] is not None:
            mask &= self._cache["diameter"] <= diameter[1]
        if approxError[0] is not None:
            mask &= self._data[:, 4] >= approxError[0]
        if approxError[1] is not None:
            mask &= self._data[:, 4] <= approxError[1]

        self._mask = mask
        return self
    

    def paint(self, image, color):
        draw = ImageDraw.Draw(image)
        for x, y, d in self._data[self._mask, :3]:
            r = d / 2
            draw.ellipse((x - r, y - r, x + r, y + r), outline = color,)

        return image


    def toOverlays(self, _unit = "px", _class = "default"):
        return [
            {
                "id": str(i) + _class,
                "type": "circle",
                "class": _class,
                "data": {
                    "x": self._data[index, 0],
                    "y": self._data[index, 1],
                    "radius": self._data[index, 2] / 2,
                },
                "tooltip": (
                    f"ID: {i}. Class: {_class}\n"
                    f"Diameter: {self._cache['diameter'][index]:.1f} {_unit}\n"
                    f"Area: {self._cache['area'][index]:.1f} {_unit}²\n"
                    f"Volume: {self._cache['volume'][index]:.1f} {_unit}³\n"
                    f"Brightness: {self._data[index, 3]:.0f}\n"
                    f"Reliability: {1 - self._data[index, 4]:.2f}"
                ),
            }
            for i, index in enumerate(np.flatnonzero(self._mask))
        ]


    

#TO DO: rework detection
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
        params = {
            "sz_med" : 3,
            "sigma_gauss": None,
            "sz_th":  4,
            "thr_br": float(settings[1]),   
            "min_dist": 3,
            "wsize": 7, 
            "best_mode": 3, 
            "msk": 3,      
            "met": 'exp',   
            "npar": 2,
            "nlocmax": 5000,
        }
    elif settings[2] == 1:
        params = {
            "sz_med" : 3,
            "sigma_gauss": 1,
            "sz_th":  6,
            "thr_br": float(settings[1]),   
            "min_dist": 3,
            "wsize": 9,
            "best_mode": 3, 
            "msk": 3,      
            "met": 'exp',   
            "npar": 2,
            "nlocmax": 1500,
        }
    elif settings[2] == 2:
        params = {
            "sz_med" : 3,
            "sigma_gauss": 1.5,
            "sz_th":  8,
            "thr_br": float(settings[1]),   
            "min_dist": 3,
            "wsize": 7,    
            "best_mode": 3,   
            "msk": 3,      
            "met": 'exp',   
            "npar": 2,
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
        return ParticleSet(np.array([]))

    if settings[2] == 2:
        blobs_appr[:, :3] *= 2
    elif settings[2] == 1 or settings[2] == 0:
        pass
    else:
        raise ValueError("!")

    return ParticleSet(
        np.column_stack([
            blobs_appr[:, 1],
            blobs_appr[:, 0],
            blobs_appr[:, 2] * 2,
            blobs_appr[:, 3],
            blobs_appr[:, 5],
        ])
    )