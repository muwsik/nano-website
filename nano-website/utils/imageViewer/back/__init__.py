import streamlit.components.v1 as components

from pathlib import Path
from PIL import Image
import io
import base64
import numpy as np


_component = components.declare_component(
    "imageViewer",
    path = str(Path(__file__).parent.parent / "front" / "dist")
)


def imageViewer(image = None, particles = None, key = None,
    metadata = {
        "unit": "px"
    }
):
    # support image type is PIL and np.ndarray
    if isinstance(image, Image.Image):
        pass
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        image = Image.fromarray(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    buffer = io.BytesIO()
    image.save(buffer, format = "PNG")
    imageBase64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")    
    width, height = image.size

    # list particles to dict
    dictParticles = []
    if particles is not None and len(particles) > 0:
        if hasattr(particles[0], "toDict"):
            dictParticles = [_particle.toDict() for _particle in particles]

    return _component(
        image = imageBase64,
        image_width = width,
        image_height = height,
        particles = dictParticles,
        key = key,
        metadata = metadata,
        default = None,
    )