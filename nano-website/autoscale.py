from PIL import Image
import numpy as np
import easyocr
import re

import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

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
        return last_white_index - first_white_index, first_white_index

    return None, None


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
        elif matchesScale[-2] == 'u' or matchesScale[-2] == 'p':
            _scale = float(matchesScale[:-2]) * 1000
    except Exception:
        _scale = None
        matchesScale = None

    return _scale, matchesScale

@st.cache_data(show_spinner = False)
def estimateScale(c_image):
    lowerBound = findBorder(c_image)
    if (lowerBound is not None):      
        text = findText(c_image[lowerBound:, :])
        scaleVal, scaleText = scale(text)
        scaleLengthVal, startPixelScale = scaleLength(c_image, lowerBound)

        if (scaleVal is not None) and (scaleLengthVal is not None):
            return scaleVal / scaleLengthVal, [lowerBound, startPixelScale, scaleLengthVal, scaleText]

    return None, None

### main
if __name__ == "__main__":    

    img_path = r"D:\Cloud\Mycroscopy\test SEM image\Pd_C_0.1%_8mm_0044_PREP_thr2_filterBr10_szm3_szth7.png"


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
        scaleVal, _ = scale(text)

        # Длина шкалы в пикселях
        scaleLengthVal, _ = scaleLength(grayImage, lowerBound)
        print(f"Длина шкалы: {scaleLengthVal} px")

        if (scaleVal is not None) and (scaleLengthVal is not None):
            print(f"nm / pixel: {scaleVal / scaleLengthVal}")
            print(f"pixel / nm: {scaleLengthVal / scaleVal}")

    _, dispScale = estimateScale(grayImage)

    x = dispScale[1]; y = dispScale[0]; length = dispScale[2]; diff = 5;
    scaleLineCoords = np.array([
        [x, y-diff], [x, y+diff], [x, y], [x+length, y], [x+length, y+diff], [x+length, y-diff]
    ])
        
    fig = px.imshow(grayImage, color_continuous_scale='gray')

    fig.add_trace(
        go.Scatter(x = scaleLineCoords[:,0], y = scaleLineCoords[:,1],
            mode='lines', line = dict(color = 'red', width = 3, dash = 'dot')
        )
    )

    fig.add_annotation(x = x + int(length/2), y = y,
        text = f"{length}px / {dispScale[3]}",
        showarrow = False,
        yshift = 40,
        font = dict(
                color = "red",
                size = 35
            )
    )

    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()
    