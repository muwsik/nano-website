import io
import csv
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

import style
import MicFunctions_v2 as mf

# Run
# streamlit run .\nano-website\nano_website.py --server.enableXsrfProtection false


# Header
style.set_style()
st.markdown("<div class='header'>WEB NANOPARTICLES</div>", unsafe_allow_html=True)
st.markdown("<div class='about'>Hello! This is a web interface for processing SEM images.</div>", unsafe_allow_html=True)

# Main content area
left, rigth = st.columns([7, 3])
uploaded_image = None

with left:
    st.header("Upload image")
    uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "tif"])

    image_placeholder = st.empty()
    if uploaded_image is not None:
        crsImage = Image.open(uploaded_image)
        image_placeholder.image(crsImage, use_column_width=True, caption="Uploaded image")

        grayCropImage = np.asarray(crsImage, dtype='uint8')[:890,:]  


with rigth:
    st.header("Settings")

    if uploaded_image is None:
        defaultQ = st.checkbox("Use default settings", value = True, disabled = True,
                               help = "You need to upload an image")
    else:
        defaultQ = st.checkbox("Use default settings", value = True)
    
    with st.container(border=True):
        if defaultQ:
            useTophatQ = st.checkbox("Use Top-hat transform", value = True, disabled = True)
            useFilterQ = st.checkbox("Use median filter", value = False, disabled = True)
            thrPrepCoef = st.text_input("Pretreatment coefficient", value="0.3", disabled = True)
        else:
            useTophatQ = st.checkbox("Use Top-hat transform", value = True)
            useFilterQ = st.checkbox("Use median filter")
            thrPrepCoef = st.text_input("Pretreatment coefficient", value="0.2")
            
    with st.container(border=True):
        if defaultQ:
            thresCoefOld = st.text_input("Nanoparticle brightness threshold", value="0.4", disabled = True)
            fsize = st.text_input("Size of the approximation window", value="7", disabled = True)         
        else:           
            thresCoefOld = st.text_input("Nanoparticle brightness threshold", value="0.5")
            fsize = st.text_input("Size of the approximation window", value="7")      

    if uploaded_image is None:
        pushProcc = st.button("Detect nanoparticles", use_container_width = True, disabled = True,
                         help = "You need to upload an image")
    else:
        pushProcc = st.button("Detect nanoparticles", use_container_width = True)
    
    # Detecting
    if pushProcc:
        currentImage = grayCropImage
        thrPrep = mf.FindThresPrep(currentImage, 1000, float(thrPrepCoef))

        if useTophatQ:            
            from skimage import morphology
            currentImage = morphology.white_tophat(currentImage, morphology.disk(4))        

        from skimage.filters import median
        filteredImage = median(currentImage, np.ones((3,3)))
        points = filteredImage > thrPrep

        if useFilterQ:
            currentImage = filteredImage
            
        image_placeholder.image(currentImage, use_column_width=True, caption="Processed image")

        radius = np.arange(1.0, 7.1, 0.1)
        BLOBs = mf.ExponentialApproximationMask(
                    currentImage, 1 / (radius ** 2), points,
                    False, int(fsize), float(thresCoefOld), 1
                )
        st.write(f"{BLOBs.shape[0]} nanoparticles found!")

        imageBLOBs = crsImage.convert("RGBA")
        draw = ImageDraw.Draw(imageBLOBs)
        for BLOB in BLOBs:                
            y, x, r = BLOB          
            draw.ellipse((x-r, y-r, x+r, y+r), outline = (0, 225, 0))
            
        image_placeholder.image(imageBLOBs, use_column_width = True, caption = "Detected nanoparticles")
    
    # Saving
    if pushProcc:
        safeImgCol, safeBLOBCol = st.columns(2)

        with safeImgCol:
            file = io.StringIO()
            #imageBLOBs.save(file, format="PNG")

            st.download_button(
                label = "Download image",
                data = file.getvalue(),
                file_name = "processed-image.tif",
                use_container_width  = True
            )

        with safeBLOBCol:
            file = io.StringIO()
            csv.writer(file).writerows(BLOBs)

            st.download_button(
                label = "Download nanoparticles",
                data = file.getvalue(),
                file_name = "nanoparticles.csv",
                use_container_width  = True
            )
    
    # Mass
    if pushProcc:
        densityPd = 12.02 * 10**-15 # nanograms / nanometer
        scale = 1; # nanometer in pixel

        radiusNM = BLOBs[:, 2] * scale;
        V = 4 / 3 * np.pi * radiusNM ** 3
        massParticles = np.sum(V * densityPd)

        st.write(f"Mass of detected Pd-nanoparticles is {massParticles:0.2e} (nanograms)")
            
st.markdown("<div class='footer'>Laboratory of Cognitive Technologies and Simulating Systems (LCTSS), Tula State University (TSU) © 2024</div>", unsafe_allow_html=True)