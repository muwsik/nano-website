import base64
import io

import numpy as np
import streamlit as st
from PIL import Image


_component = st.components.v2.component(
    "streamlit-image-viewer.streamlit_image_viewer",
    js="index-*.js",
    html='<div class="react-root"></div>',
)


def streamlit_image_viewer(
    image = None,
    particles = None,
    key = None,
    metadata = {"unit": "px"},
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
    image.save(buffer, format="PNG")
    width, height = image.size

    imageBase64 = (
        "data:image/png;base64,"
        + base64.b64encode(buffer.getvalue()).decode("utf-8")
    )

    dictParticles = []
    if particles is not None and len(particles) > 0:
        if hasattr(particles[0], "toDict"):
            dictParticles = [
                particle.toDict()
                for particle in particles
            ]

    return _component(
        key = key,
        data = {
            "image": imageBase64,
            "imageWidth": width,
            "imageHeight": height,
            "particles": dictParticles,
            "metadata": metadata,
        },
        default = None
    )