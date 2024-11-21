import streamlit as st

import MicFunctions_v2 as mf

@st.cache_data(show_spinner = False)
def CACHE_FindThresPrep(img, nbr, thrPrepCoef):
    return mf.FindThresPrep(img, nbr, thrPrepCoef)

@st.cache_data(show_spinner = "Nanoparticle detection on process...")
def CACHE_ExponentialApproximationMask(img, c1s, points, maskMain, wsize, thresCoefOld, nproc):
    return mf.ExponentialApproximationMask(img, c1s, points, maskMain, wsize, thresCoefOld, nproc)
