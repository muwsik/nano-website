from pathlib import Path
import io
from types import SimpleNamespace


_EXAMPLES = [
    {
        "name": "ED 779-A3_0009.tif",
        "type": 1,  # SEM
        "source": "https://doi.org/10.6084/m9.figshare.11783661.v1",
        "particles": 200,
        "time": 5,

        "d_brightness": 5,
        "d_diameter": 2,    # Large (10-30 pixels)
        "f_brightness": 3,
        "f_diameters": (7.6, 12.2),
        "f_reliability": 0.83,
    },
    {
        "name": "Pd-1.jpg",
        "type": 2,      # TEM
        "source": None,
        "particles": 4500,
        "time": 50,

        "d_brightness": 15,
        "d_diameter": 0,    # Small (1-10 pixels)
        "f_brightness": 15,
        "f_diameters": (1.0, 11.0),
        "f_reliability": 0.75,
    },
    {
        "name": "Pd-PDBSE-1.tif",
        "type": 1, # SEM
        "source": None,
        "particles": 900,
        "time": 15,

        "d_brightness": 20,
        "d_diameter": 1,    # Medium (5-15 pixels)
        "f_brightness": 25,
        "f_diameters": (9.5, 28.0),
        "f_reliability": 0.84,
    },
]


class ExampleFile(io.BytesIO):
    def __init__(self, path, title):
        with open(path, "rb") as file:
            super().__init__(file.read())

        self.name = path.name
        self.title = title

        match path.suffix.lower():
            case ".tif" | ".tiff":
                self.type = "image/tiff"
            case ".png":
                self.type = "image/png"
            case ".jpg" | ".jpeg":
                self.type = "image/jpeg"
            case _:
                self.type = "application/octet-stream"


def createExamples():
    examples = []
    for example in _EXAMPLES:
        source = f"[**Source**]({example['source']})" if example["source"] else ""

        title = (
            f"**{example['type']}**-image with ~**{example['particles']}** particles. "            
            f"Work time: ~{example['time']} s. "
            f"{source}"
        )

        examples.append(ExampleFile(
            Path("./nano-website/content/EM-examples") / example["name"],
            title
        ))

    return examples


def getExampleParams(name):    
    for example in _EXAMPLES:
        if example["name"] == name:
            return SimpleNamespace(
                type = example["type"],
                d_brightness = example["d_brightness"],
                d_diameter = example["d_diameter"],
                f_brightness = example["f_brightness"],
                f_diameter = example["f_diameters"],
                f_reliability = example["f_reliability"],
            )


    return None