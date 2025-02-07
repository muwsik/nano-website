import streamlit as st
import MicFunctions_v2 as mf
import random
import numpy

@st.cache_data(show_spinner = False)
def CACHE_FindThresPrep(img, nbr, thrPrepCoef):
    return mf.FindThresPrep(img, nbr, thrPrepCoef)

@st.cache_data(show_spinner = "Nanoparticle detection on process...")
def CACHE_ExponentialApproximationMask(img, c1s, points, maskMain, wsize, thresCoefOld, nproc):
    return mf.ExponentialApproximationMask(img, c1s, points, maskMain, wsize, thresCoefOld, nproc)

def randon_BLOBS(count = 250, x_max = 1200, y_max = 900):
    fake_BLOBS = numpy.zeros((count, 3))

    for i in range(len(fake_BLOBS)):
        fake_BLOBS[i, 0] = random.randint(0, y_max)
        fake_BLOBS[i, 1] = random.randint(0, x_max)
        fake_BLOBS[i, 2] = random.uniform(0, 7)

    return fake_BLOBS