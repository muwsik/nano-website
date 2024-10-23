from PIL import Image
import numpy as np
import easyocr
import cv2
import os
import re

import matplotlib.pyplot as plt

# Дополнительные функции, необходимые для обработки изображений
def findBorder(_fullImage, thr = 0.5):    
    row_sum = np.sum(_fullImage, axis = 1, dtype = np.int64)

    for i in range(len(row_sum)):
        if np.abs(row_sum[i] - row_sum[i + 1]) >= row_sum[i] * thr:
            return i + 1
    
    return None

def scaleLength(_fullImage, start_y):
    height, width = _fullImage.shape
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

def findText(img):
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    result = reader.readtext(img, detail=0, blocklist='SOo')
    return ' '.join(result)  

def increase(_text):
    try:
        matchesIncrease = re.findall(r'[xX][0-9]*.?[0-9]+[kK]', _text)[0]
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



### main


img_path = r"D:\Cloud\Mycroscopy\gt\EP-483_i042\EP-483_i042.tif"
#img_path = r"C:\Users\Muwa\Desktop\2024-09-10-Ivanova\Pd_C_0.1%_8mm\Pd_C_0.1%_8mm_0018.tif"


img = Image.open(img_path)
grayImage = np.array(img, dtype='uint8')

# Высота только изображения (без нижней сноски)
lowerBound = findBorder(grayImage)

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

if (scaleVal is not None) and (scaleLengthVal is not None):
    print(f"nm / pixel: {scaleVal / scaleLengthVal}")
    print(f"pixel / nm: {scaleLengthVal / scaleVal}")


