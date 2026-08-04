from streamlit_card import card

import base64, io       
from pathlib import Path
from PIL import Image

from content.tooltips import Options


_EXAMPLES = [
    {
        "id": "ex1",
        "imageName": "ED 779-A3_0009.tif",
        "title": "SEM-image",
        "particles": 200,
        "time": 5,

        "d_brightness": 5,
        "d_diameter": 2,
        "f_brightness": 3,
        "f_diameters": (7.6, 12.2),
        "f_reliability": 0.83,
    },
    {
        "id": "ex2",
        "imageName": "Pd-1.jpg",
        "title": "TEM-image",
        "particles": 4500,
        "time": 50,

        "d_brightness": 15,
        "d_diameter": 0,
        "f_brightness": 15,
        "f_diameters": (1.0, 11.0),
        "f_reliability": 0.75,
    },

]

def getExample(imageID):
    for _i in _EXAMPLES:
        if _i['id'] == imageID:
            buffer = io.BytesIO()
            Image.open(Path(__file__).parent / _i['imageName']).save(buffer, format = "PNG")
            data = (
                "data:image/png;base64,"
                + base64.b64encode(buffer.getvalue()).decode("utf-8")
            )

            return card(
                title = f"{_i['title']} with ~{_i['particles']} particles",
                text = [
                    f"""Detection params: brightness = {_i['d_brightness']};
                        size = {Options.NanopartSize[_i['d_diameter']]}. Time: {_i['time']} s""",
                    f"""Filtration params: brightness = {_i['f_brightness']};
                        diameters = {_i['f_diameters']} nm; reliability = {_i['f_reliability']}"""
                ],
                image = data,
                # url = "image source link"
                styles = {
                    "card": {
                        "width": "620px",
                        "height": "480px",
                        "justify-content": "flex-end",
                            "padding": "0 18px 18px 18px",
                    },
                    "filter": {
                        "background": (
                            "linear-gradient("
                                "to top,"
                                "rgba(0,0,0,0.85) 20%,"
                                "rgba(0,0,0,0.1) 25%,"
                                "rgba(0,0,0,0) 100%"
                            ")"
                        )
                    },
                    "title": {
                        "font-size": "25px",
                    },
                    "text": {
                        "font-size": "13px",
                    },
                    "div": {
                        "margin": "0px",
                    }
                }
            )

    return None
