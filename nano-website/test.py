import os
from PIL import Image
import easyocr
import re
import cv2
import numpy as np

def process_file(img):
    dist = distance(img, black_pixels(img))
    rez = text_out(img)
    text_result = ' '.join(rez)               
    uvel, shkal = extract_values(text_result)

    print(f"Длинна шкилы: {dist}")
    print(f"Увеличение: {uvel}")
    print(f"Число под шкалой: {shkal}")
    if shkal is not None:
        fraction = dist / shkal
        print(f"Дробь: {fraction}")


# Дополнительные функции, необходимые для обработки изображений
def black_pixels(img):    
    width, height = img.size
    last_black_y = -1
    
    for y in range(height - 1, -1, -1):
        pixel = img.getpixel((width - 1, y))
        if pixel == (0, 0, 0):
            last_black_y = y
        else:
            break
    print(f"Координата X: {width - 1}, Координата Y: {last_black_y}")
    
    crop_img = img.crop((0, last_black_y, width, height))    
    
    return last_black_y

def distance(img, start_y):
    width, height = img.size
    first_white_index = None
    last_white_index = None

    for x in range(width):
        pixel = img.getpixel((x, start_y))
        if pixel == (255, 255, 255):
            if first_white_index is None:
                first_white_index = x
            last_white_index = x

    if first_white_index is not None and last_white_index is not None:
        return last_white_index - first_white_index
    return 0

def text_out(img):
    img_np = np.array(img)

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    result = reader.readtext(img_np, detail=0, blocklist='SOo')
    return result

def extract_values(text):
    matches_increase = re.findall(r'[xX][0-9]{3}[kK]', text)[0]
    try:
        increase = int(matches_increase[1:-1])
    except Exception:
        increase = None


    matches_scale = re.findall(r'\d+.\d{0,3}[nup]m', text)[0]
    try:
        if matches_scale[-2] == 'n':
            scale = int(matches_scale[:-2])
        if matches_scale[-2] == 'u' or matches_scale[-2] == 'p':
            scale = float(matches_scale[:-2]) * 1000
    except Exception:
        scale = None

    return increase, scale

img_path = r"D:\Cloud\Mycroscopy\gt\EP-483_i042\EP-483_i042.tif"


img = Image.open(img_path)
img = img.convert('RGB')

process_file(img)
