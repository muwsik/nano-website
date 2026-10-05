from PIL import Image

from content.tooltips import Options

EXAMPLES = [
    {
        "id": "ex1",
        "imageName": "ED 779-A3_0009.tif",
        "type": "SEM",
        "source": "https://doi.org/10.6084/m9.figshare.11783661.v1",
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
        "type": "TEM",
        "source": None,
        "particles": 4500,
        "time": 50,

        "d_brightness": 15,
        "d_diameter": 0,
        "f_brightness": 15,
        "f_diameters": (1.0, 11.0),
        "f_reliability": 0.75,
    },
    {
        "id": "ex3",
        "imageName": "Pd-PDBSE-1.tif",
        "type": "TEM",
        "source": None,
        "particles": 900,
        "time": 15,

        "d_brightness": 20,
        "d_diameter": 1,
        "f_brightness": 25,
        "f_diameters": (9.5, 28.0),
        "f_reliability": 0.84,
    },
]

def loadExamples():
    examples = []

    for example in EXAMPLES:
        path = f"nano-website/content/EM-examples/{example['imageName']}"
        with Image.open(path) as img:
            image = img.copy()

        source = f"[**Source**]({example['source']})" if example["source"] else ""

        examples.append({
            "id": example["id"],
            "image": image,
            "title": (
                f"**{example['type']}**-image with ~**{example['particles']}** particles. "                      
                f"{source}"
            ),
            "description": (
                f"**Detection** → "
                f"Brightness: {example['d_brightness']}; "
                f"Size: {Options.NanopartSize[example['d_diameter']]}. "
                f"Time: ~{example['time']} s  \n\n"

                f"**Filtration** → "
                f"Brightness: {example['f_brightness']}; "
                f"Diameter: {example['f_diameters'][0]}–{example['f_diameters'][1]} nm; "
                f"Reliability: {example['f_reliability']:.2f}"
            ),             
        })

    return examples