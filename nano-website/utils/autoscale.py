from PIL import Image
import numpy as np
import easyocr
import re

def findBorder(c_fullImage, thr = 0.5):    
    row_sum = np.sum(c_fullImage, axis = 1, dtype = np.int64)

    for i in range(len(row_sum) - 1):
        if np.abs(row_sum[i] - row_sum[i + 1]) >= row_sum[i] * thr:
            return i + 1
    
    return None

def scaleLength(borderLine, threshBin = 128):
    binLine = (borderLine > threshBin).astype(np.uint8)

    diffLine = np.diff(binLine)

    indices = np.where(diffLine == 1)[0] + 1    # +1 for the ordinal number

    if indices.size:
        return indices[-1] - indices[0], indices[0]

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
        matchesScale = re.findall(r"[0-9]*\.?[0-9]+(?:nm|um|pm|pum)", c_text)[0]
        if matchesScale[-2] == 'n':
            _scale = float(matchesScale[:-2])
        elif matchesScale[-2] == 'u' or matchesScale[-2] == 'p':
            if matchesScale[-3] == '0':
                _scale = float(matchesScale[:-2]) * 1000
            else:
                _scale = float(matchesScale[:-3]) * 1000
    except Exception:
        _scale = None
        matchesScale = None

    return _scale, matchesScale


def analyzeScaleRegion(c_image):
    if isinstance(c_image, np.ndarray):
        pass
    elif isinstance(c_image, Image.Image):
        c_image = np.array(c_image, dtype = 'uint8')
    else:
        raise ValueError("!")

    lowerBound = findBorder(c_image)
    if (lowerBound is not None):      
        text = findText(c_image[lowerBound:, :])
        scaleVal, scaleText = scale(text)
        scaleLengthVal, startPixelScale = scaleLength(c_image[lowerBound])

        if (scaleVal is not None) and (scaleLengthVal is not None):
            return scaleVal / scaleLengthVal, lowerBound, [startPixelScale, scaleLengthVal, scaleText]
     
    return None, lowerBound, None


class Scale:
    def __init__(self, multiplier = None):

        if (multiplier is None):
            self._mode = 'PIXELS'
            self._multiplier = 1.0
        else:            
            self.setScale(multiplier)
                

    def setScale(self, multiplier):
        if multiplier <= 0:
            raise ValueError("Scale must be positive.")

        self._mode = 'METRIC'
        self._multiplier = multiplier    
        

    def apply(self, value):
        return np.asarray(value) * self._multiplier


    @property
    def unit(self):
         return "nm" if self._mode == 'METRIC' else "px"    
     
    @property
    def multiplier(self):
        return self._multiplier

    @property
    def divider(self):
        return 1 / self._multiplier
     


### main
if __name__ == "__main__":  
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    img_path = r"C:\Users\Muwa\Desktop\2-S1-no_area-100k-ordered (6).tif"

    img = Image.open(img_path).convert('L')
    img = img.resize((1280, 960))
    grayImage = np.array(img, dtype='uint8')

    tmp = Scale(grayImage)

    print(tmp.__dict__)
