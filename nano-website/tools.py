#import cv2
#import pytesseract
#import re
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#def prepair_img(img, crop_coef=800):
#    img = img[crop_coef:, :]
#    ret, thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)
#    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#    return contours
#def recognize_and_crop(contours, img):
#    im2 = img.copy()
#    height = im2.shape[0]
#    width = im2.shape[1]
#    recognized_text = ""
#    cropped = im2
#    for cnt in contours:
#        x, y, w, h = cv2.boundingRect(cnt)
#        if h >= int(0.05 * height) and w >= int(1.0 * width):
#            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
#            cropped = im2[y:y + h, x:x + w]
#            text = pytesseract.image_to_string(cropped)
#            if text != "":
#                recognized_text += text
#    if recognized_text == "":
#        exit(1)
#    scale_x = re.findall(r'x\d+.\d{0,3}', recognized_text)
#    #print(scale_x)
#    if len(scale_x) == 0:
#        exit(1)
#    scale_x = float(scale_x[0][1:-1])
#    length_nm = re.findall(r'\d+.\d{0,3}nm', recognized_text)
#    if len(length_nm) == 0:
#        length_um = re.findall(r'\d+.\d{0,3}[up]m', recognized_text)
#        if len(length_um) == 0:
#            exit(1)
#        length_um = length_um[0][:-2]
#        length_nm = float(length_um) * 1000
#    else:
#        length_nm = length_nm[0][:-2]
#    return scale_x, int(length_nm), cropped

import streamlit as st
import MicFunctions_v2 as mf

@st.cache_data(show_spinner = False)
def CACHE_FindThresPrep(img, nbr, thrPrepCoef):
    return mf.FindThresPrep(img, nbr, thrPrepCoef)

@st.cache_data(show_spinner = "Nanoparticle detection is underway...")
def CACHE_ExponentialApproximationMask(img, c1s, points, maskMain, wsize, thresCoefOld, nproc):
    return mf.ExponentialApproximationMask(img, c1s, points, maskMain, wsize, thresCoefOld, nproc)