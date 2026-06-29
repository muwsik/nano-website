import streamlit as st # for st.cache_data

import warnings
import numpy as np

class Particle:
    def __init__(self, centerCoords, diametr):
        self.x = centerCoords[0]
        self.y = centerCoords[1]

        if (diametr > 0):
            self.diameter = diametr
        else:
            warnings.warn("The particle diameter is less than zero!")            
            self.diameter = 0

        self.projectionArea = 1 / 4 * np.pi * self.diameter**2 

        self.volume = 2 / 3 * self.projectionArea * self.diameter

        # detection features 
        self.c0 = None
        self.approxError = None

    def toDict(self):
        return self.__dict__.copy()


def blobs2Particles(blobs):
    result = []

    for temp in blobs:
        new = Particle((temp[1], temp[0]), temp[2]*2)

        new.c0 = temp[3]
        new.approxError = temp[5]

        result.append(new)

    return np.array(result)


def inRange(value, limits):
    min_val, max_val = limits

    if min_val is not None and value < min_val:
        return False
    if max_val is not None and value > max_val:
        return False

    return True


@st.cache_data(show_spinner = False, max_entries = 5)
def filtrationParticles(particles, 
    c0 = (None, None), 
    diameter = (None, None), 
    approxError = (None, None)
):
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