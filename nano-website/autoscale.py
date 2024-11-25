import streamlit as st

from PIL import Image
import glob, os
import cv2
import numpy as np
import easyocr
import re

import matplotlib.pyplot as plt


@st.cache_data(show_spinner = False)
def findBorder(_fullImage, thr = 0.5):    
    row_sum = np.sum(_fullImage, axis = 1, dtype = np.int64)

    for i in range(len(row_sum) - 1):
        if np.abs(row_sum[i] - row_sum[i + 1]) >= row_sum[i] * thr:
            return i + 1
    
    return None

@st.cache_data(show_spinner = False)
def scaleLength(_fullImage, start_y):
    __, width = _fullImage.shape
    first_white_index = None
    last_white_index = None

    for x in range(1, width):
        if (_fullImage[start_y, x] == 255) and (_fullImage[start_y, x-1] == 0):
            if first_white_index is None:
                first_white_index = x
            last_white_index = x

    if first_white_index is not None and last_white_index is not None:
        return last_white_index - first_white_index

    return None

@st.cache_data(show_spinner = False)
def findText(_footnoteImage):
    reader = easyocr.Reader(["en"], gpu = False, verbose = False)
    result = reader.readtext(_footnoteImage, detail = 0, blocklist = 'SOo')
    return ' '.join(result).lower()  

def increase(_text):
    try:
        matchesIncrease = re.findall(r'[x][0-9]*.?[0-9]+[k]', _text)[0]
        _increase = float(matchesIncrease[1:-1])
    except Exception:
        _increase = None

    return _increase

def scale(_text):
    try:
        matchesScale = re.findall(r'[0-9]*.?[0-9]+[nup]m', _text)[0]
        if matchesScale[-2] == 'n':
            _scale = float(matchesScale[:-2])
        if matchesScale[-2] == 'u' or matchesScale[-2] == 'p':
            _scale = float(matchesScale[:-2]) * 1000
    except Exception:
        _scale = None

    return _scale


@st.cache_data(show_spinner = False)
def load_templates():
    files = glob.glob(r"./template/*.tif")
    print(os.getcwd())
    print(files)
    templates = []
    for file in files:
        str_scale = file.split('/')[-1].split('.')[0]
        templates.append([str_scale, np.array(Image.open(file).convert('L'), dtype = 'uint8')])

    return templates


def scale_template(_footnoteImage, _thr = 0.5):
    templates = load_templates()
        
    matchingVal = []
    for str, template in templates:
        tempMatching = cv2.matchTemplate(_footnoteImage, template, method = cv2.TM_SQDIFF_NORMED)     
        tempMinVal, _, _, _ = cv2.minMaxLoc(tempMatching)
        matchingVal.append(tempMinVal)
        st.write(tempMinVal)
        
    if np.min(matchingVal) <= _thr:
        return scale(templates[np.argmin(matchingVal)][0])

    return None


### main
if __name__ == "__main__":    

    img_path = r"C:\Users\Muwa\Downloads\11783661\497-S1-A62-200k-ordered.tif"


    img = Image.open(img_path).convert('L')
    grayImage = np.array(img, dtype='uint8')

    # Высота только изображения (без нижней сноски)
    lowerBound = findBorder(grayImage)
    print(f"Граница: {lowerBound} px")
    
    if not (lowerBound is None):
        temp = scale_template(grayImage[lowerBound:, :])
        print(f"Масштаб v2: {temp}")


    if not (lowerBound is None):
        # Сноска
        #plt.imshow(grayImage[lowerBound:, :])

        # Только изображение
        #plt.imshow(grayImage[:lowerBound, :])

        # Распознавание текста в сноске
        text = findText(grayImage[lowerBound:, :])
        print("Текст:", text)

        # Увеличение
        increaseVal = increase(text)
        print(f"Увеличение: {increaseVal}")

        # Длина шкалы в нанометрах
        scaleVal = scale(text)
        print(f"Масштаб: {scaleVal}")

        # Длина шкалы в пикселях
        scaleLengthVal = scaleLength(grayImage, lowerBound)
        print(f"Длина шкалы: {scaleLengthVal} px")

        if (scaleVal is not None) and (scaleLengthVal is not None):
            print(f"nm / pixel: {scaleVal / scaleLengthVal}")
            print(f"pixel / nm: {scaleLengthVal / scaleVal}")


