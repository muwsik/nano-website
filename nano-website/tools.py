import streamlit as st
import MicFunctions_v2 as mf2
import MicFunctions_v3 as mf3

import numpy as np


#@st.cache_data(show_spinner = False)
#def CACHE_FindThresPrep(img, nbr, thrPrepCoef):
#    return mf2.FindThresPrep(img, nbr, thrPrepCoef)

#@st.cache_data(show_spinner = "Nanoparticle detection on process...")
#def CACHE_ExponentialApproximationMask(img, c1s, points, maskMain, wsize, thresCoefOld, nproc):
#    return mf2.ExponentialApproximationMask(img, c1s, points, maskMain, wsize, thresCoefOld, nproc)


@st.cache_data(show_spinner = False)
def CACHE_HelpMatricesNew(_wsize, _rs):
    return mf3.MakeHelpMatricesNew(_wsize, _rs)


@st.cache_data(show_spinner = False)
def CACHE_PrefilteringPoints(_img, _sz_med, _sz_th, _min_dist, _thr_br):
    img_med = mf3.PreprocessingMedian(_img, _sz_med)
    img_med_th = mf3.PreprocessingTopHat(img_med, _sz_th) 
    img_th = mf3.PreprocessingTopHat(_img, _sz_th)
    
    lm, nlmax = mf3.PrefilteringPoints(img_med_th, _min_dist, _thr_br)

    return lm

#@st.cache_data(show_spinner = "Nanoparticle detection on process...")
def CACHE_ExponentialApproximationMask_v3(_img, _lm, _xy2, _helpMatrs, _params, _prn = False):
    number_blobs = len(_lm)

    blobs_full = np.zeros([number_blobs, 3])      # blobs_full[i] = y, x, r
    values_full = np.zeros([number_blobs, 4])     # values_full[i] = c0, c1, c2, norm_error

    for i, temp_lm in enumerate(_lm):
        blob, c0, c1, c2, norm_error = mf3.ApproximationWithFindingTheBestCenter_NoFiltering(
            _img,    
            temp_lm,
            _xy2,
            _helpMatrs,
            _params,
            _prn
        )

        blobs_full[i, :] = blob
        values_full[i, :] = c0, c1, c2, norm_error

    return blobs_full, values_full


def my_FilterBlobs(_blobs_ext, _blobs_param, _params):
    return np.array(mf3.FilterBlobs_change(_blobs_ext, _blobs_param, _params)[0])