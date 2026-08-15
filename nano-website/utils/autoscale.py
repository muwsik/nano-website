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


def scale(c_text):
    try:
        matchesScale = re.findall(r"\b\d+(?:\.\d+)?(?:nm|um|pm|pum)\b", c_text)[0]
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


def detectScale(image):
    if isinstance(image, Image.Image):
        image = np.array(image, dtype = np.uint8)
    elif not isinstance(image, np.ndarray):
        raise ValueError("Image must be a NumPy array or PIL Image.")

    horizontalBound = findBorder(image)

    if horizontalBound is None:
        return Scale()

    fullText = findText(image[horizontalBound:, :])

    scaleBarVal, scaleBarText = scale(fullText)

    scaleBarLength, scaleBarX = scaleLength(image[horizontalBound])

    if scaleBarVal is None or scaleBarLength is None:
        return Scale()

    return Scale(
        multiplier = scaleBarVal / scaleBarLength,
        info = [
            horizontalBound,
            scaleBarX,
            scaleBarLength,
            scaleBarText
        ]
    )


class Scale:
    def __init__(self, multiplier = None, info = None):
        if multiplier is None:
            self._mode = "PIXELS"
            self._multiplier = 1.0
        else:
            self.setScale(multiplier)

        self._info = info

    def setScale(self, multiplier, info = None):
        if multiplier <= 0:
            raise ValueError("Scale must be positive.")

        self._mode = "METRIC"
        self._multiplier = multiplier
        self._info = info

    def apply(self, value):
        return np.asarray(value) * self._multiplier

    @property
    def unit(self):
        return "nm" if self._mode == "METRIC" else "px"

    @property
    def multiplier(self):
        return self._multiplier

    @property
    def divider(self):
        return 1 / self._multiplier

    @property
    def info(self):
        return self._info

    @property
    def horizontalBound(self):
        return self._info[0] if self._info is not None else None
     


### main
if __name__ == "__main__":  

    img_path = r"C:\Users\Victory\Downloads\fMaIVQuFjZcJ1_OC-7l0q4YstN3SSGwjMOr9EdOl0G74L66EMNPJbnflPmVsPcLsNj56NQEQ5-pUvKW6dpvy_0dQ.jpg"

    img = Image.open(img_path).convert('L')
    img = img.resize((1280, 960))
    grayImage = np.array(img, dtype='uint8')

    tmp = detectScale(grayImage)

    print(tmp.__dict__)
