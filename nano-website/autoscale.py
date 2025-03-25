from PIL import Image
import numpy as np
import easyocr
import re

import streamlit as st

def findBorder(c_fullImage, thr = 0.5):    
    row_sum = np.sum(c_fullImage, axis = 1, dtype = np.int64)

    for i in range(len(row_sum) - 1):
        if np.abs(row_sum[i] - row_sum[i + 1]) >= row_sum[i] * thr:
            return i + 1
    
    return None

def scaleLength(c_fullImage, start_y):
    _, width = c_fullImage.shape
    first_white_index = None
    last_white_index = None

    for x in range(1, width):
        if (c_fullImage[start_y, x] >= 230 ) and (c_fullImage[start_y, x-1] <= 25):
            if first_white_index is None:
                first_white_index = x
            last_white_index = x

    if first_white_index is not None and last_white_index is not None:
        return last_white_index - first_white_index

    return None


def findText(c_footnoteImage):
    reader = easyocr.Reader(["en"], gpu = False, verbose = False)
    result = reader.readtext(c_footnoteImage, detail = 0, blocklist = 'SOo')
    return ' '.join(result).lower()  

def increase(c_text):
    try:
        matchesIncrease = re.findall(r'[x][0-9]*\.?[0-9]+[k]', c_text)[0]
        _increase = float(matchesIncrease[1:-1])
    except Exception:
        _increase = None

    return _increase

def scale(c_text):
    try:
        matchesScale = re.findall(r"[0-9]*\.?[0-9]+[nup]m", c_text)[0]
        if matchesScale[-2] == 'n':
            _scale = float(matchesScale[:-2])
        if matchesScale[-2] == 'u' or matchesScale[-2] == 'p':
            _scale = float(matchesScale[:-2]) * 1000
    except Exception:
        _scale = None

    return _scale

@st.cache_data(show_spinner = False)
def estimateScale(c_image):
    lowerBound = findBorder(c_image)
    if (lowerBound is not None):      
        text = findText(c_image[lowerBound:, :])
        scaleVal = scale(text)
        scaleLengthVal = scaleLength(c_image, lowerBound)

        if (scaleVal is not None) and (scaleLengthVal is not None):
            return scaleVal / scaleLengthVal
        else:
            return None

### main
if __name__ == "__main__":    

    img_path = r"C:\Users\Muwa\Desktop\test-image.tif"


    img = Image.open(img_path).convert('L')
    grayImage = np.array(img, dtype='uint8')

    # Высота только изображения (без нижней сноски)
    lowerBound = findBorder(grayImage)
    print(f"Граница: {lowerBound} px")

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

        # Длина шкалы в пикселях
        scaleLengthVal = scaleLength(grayImage, lowerBound)
        print(f"Длина шкалы: {scaleLengthVal} px")

        if (scaleVal is not None) and (scaleLengthVal is not None):
            print(f"nm / pixel: {scaleVal / scaleLengthVal}")
            print(f"pixel / nm: {scaleLengthVal / scaleVal}")

    print(estimateScale(grayImage))