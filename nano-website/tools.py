import streamlit as st
import MicFunctions_v2 as mf
import random
import numpy as np

@st.cache_data(show_spinner = False)
def CACHE_FindThresPrep(img, nbr, thrPrepCoef):
    return mf.FindThresPrep(img, nbr, thrPrepCoef)

@st.cache_data(show_spinner = "Nanoparticle detection on process...")
def CACHE_ExponentialApproximationMask(img, c1s, points, maskMain, wsize, thresCoefOld, nproc):
    return mf.ExponentialApproximationMask(img, c1s, points, maskMain, wsize, thresCoefOld, nproc)

def randon_BLOBS(count = 250, x_max = 1200, y_max = 890):
    fake_BLOBS = np.zeros((count, 3))

    for i in range(len(fake_BLOBS)):
        fake_BLOBS[i, 0] = random.randint(0, y_max)
        fake_BLOBS[i, 1] = random.randint(0, x_max)
        fake_BLOBS[i, 2] = random.uniform(0, 7)

    return fake_BLOBS


def uniformity(BLOBs, sizeImage, sizeBlock):
    heightBlocks = int(np.ceil(sizeImage[0] / sizeBlock))
    widthBlocks = int(np.ceil(sizeImage[1] / sizeBlock))
    
    counter = np.zeros((heightBlocks, widthBlocks), dtype = int)        
    for blob in BLOBs:
        y, x, r = blob
        i = np.ceil(y / sizeBlock) - 1
        j = np.ceil(x / sizeBlock) - 1
        counter[int(i), int(j)] += 1

    return counter