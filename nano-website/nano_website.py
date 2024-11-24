import streamlit as st

import io, csv
import numpy as np
from skimage import morphology
from skimage.filters import median
from streamlit_image_comparison import image_comparison
from PIL import Image, ImageDraw
    
import style, tools, autoscale

# Run
# streamlit run .\nano-website\nano_website.py --server.enableXsrfProtection false

help_str = "be added soon"


def load_default_settings():
    st.session_state['imageUpload'] = False
    st.session_state['uploadedImage'] = None

    st.session_state['settingDefault'] = True

    st.session_state['detected'] = False
    st.session_state['BLOBs'] = None
    st.session_state['imageBLOBs'] = None

    st.session_state['comparison'] = False


if 'imageUpload' not in st.session_state:
    load_default_settings()


# Header
style.set_style()
st.markdown("<div class = 'header'>WEB NANOPARTICLES</div>", unsafe_allow_html = True)
st.markdown("""<div class = 'about'>
                    Hello! It is an interactive tool for processing images from a scanning electron microscope (SEM).
                    It will help you to detect palladium nanoparticles in the image and calculate their mass.
                    Examples of SEM images for analysis are <a href=https://doi.org/10.6084/m9.figshare.11783661.v1>here</a>.
               </div>""", unsafe_allow_html = True)
st.markdown("""<div class = 'cite'> <b>How to cite</b>: Mikhail Yu. Kurbakov, Valentina V. Sulimova, Andrei V. Kopylov,
                    Oleg S. Seredina, Alexey S. Kashin, Nina M. Ivanova, Valentine P. Ananikov. The article. Journal. 2025.
               </div>""", unsafe_allow_html = True)


# Main content area
left, rigth = st.columns([9, 3])


with left:
    st.header("Upload SEM image")
    uploadedImage = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "tif"])
    
    imagePlaceholder = st.empty()

    if uploadedImage is None:        
         load_default_settings()
    else:
        st.session_state['imageUpload'] = True
        crsImage = Image.open(uploadedImage)
        grayImage = np.array(crsImage.convert('L'), dtype = 'uint8')
               

        if (not np.array_equal(st.session_state['uploadedImage'], grayImage)):
            st.session_state['uploadedImage'] = grayImage
            st.session_state['detected'] = False

        if (not st.session_state['detected']):
            imagePlaceholder.image(crsImage, use_column_width = True, caption = "Uploaded image")
        elif (not st.session_state['comparison']):
            imagePlaceholder.image(st.session_state['imageBLOBs'], use_column_width = True, caption = "Detected nanoparticles")        
        else:
            st.markdown(
                f"""
                    <style>
                    iframe {{
                        width: inherit;
                        height: 1150px;
                    }}
                    </style>
                """, unsafe_allow_html = True)

            image_comparison(
               img1 = crsImage,
               img2 = st.session_state['imageBLOBs'],
               label1 = "Initial SEM image",
               label2 = "Detected nanoparticles",
               in_memory = True)


with rigth:
    st.header("Settings")

    st.checkbox("Use default settings?",
        disabled = not st.session_state['imageUpload'],
        key = 'settingDefault',
        help = "You need to upload an SEM image")
    
    # Preprocessing settings
    with st.container(border = True):
        methodPrep = st.selectbox("Image preprocessing method",
             ("None", "Top-hat  &  threshold filtering", "New!"),
             index = 1 if st.session_state['settingDefault'] else 0,
             disabled = st.session_state['settingDefault'],
             help = help_str)

        if methodPrep == "Top-hat  &  threshold filtering":
            thrPrepCoef = st.text_input("Pretreatment filtering coefficient",
                value = "0.2" if st.session_state['settingDefault'] else "0.3",
                disabled = st.session_state['settingDefault'],
                help = help_str)

        dispCurrentImage = st.checkbox("Display image after preprocessing?",
            value = False if st.session_state['settingDefault'] else True,
            disabled = st.session_state['settingDefault'],
            help = help_str)
    
    # Nanoparticle detection settings
    with st.container(border = True):        
        methodDetect = st.selectbox("Image preprocessing method",
             ("Exponential approximation", "New!"),
             index = 0 if st.session_state['settingDefault'] else 0,
             disabled = st.session_state['settingDefault'],
             help = help_str)

        if methodDetect == "Exponential approximation":
            thresCoefOld = st.text_input("Nanoparticle brightness filtering threshold",
                value = "0.5" if st.session_state['settingDefault'] else "0.4",
                disabled = st.session_state['settingDefault'],
                help = help_str)

            fsize = st.selectbox("Size of the approximation window (in pixels)",
                 (5, 7, 9, 11, 13),
                 index = 1 if st.session_state['settingDefault'] else 0,
                 disabled = st.session_state['settingDefault'],
                 help = help_str)


    pushProcc = st.button("Nanoparticles detection ",
        use_container_width = True,
        disabled = not st.session_state['imageUpload'],
        help = "You need to upload an SEM image")
    
    # Detecting
    if pushProcc:
        currentImage = np.copy(grayImage) 
         
        lowerBound = autoscale.findBorder(grayImage)
        #print(f"Граница: {lowerBound} px")
        
        if (lowerBound is not None):
            currentImage = currentImage[:lowerBound, :]

        if methodPrep != "None":
            # Adaptive threshold
            thrPrep = tools.CACHE_FindThresPrep(currentImage, 1000, float(thrPrepCoef)) 

            # Top-hat
            currentImage = morphology.white_tophat(currentImage, morphology.disk(4))
        
            # Pre-filtering
            filteredImage = median(currentImage, np.ones((3,3)))
            approxPoint = filteredImage > thrPrep
        else:
            approxPoint = np.ones_like(currentImage)
        
        if dispCurrentImage:
            imagePlaceholder.image(currentImage, use_column_width = True, caption = "Processed image")

        # Approximation 
        radius = np.arange(1.0, 7.1, 0.1)
        BLOBs = tools.CACHE_ExponentialApproximationMask(
                    currentImage, 1 / (radius ** 2), approxPoint,
                    False, int(fsize), float(thresCoefOld), 3)

        st.session_state['BLOBs'] = BLOBs
        st.session_state['detected'] = True

        imageBLOBs = crsImage.convert("RGBA")
        draw = ImageDraw.Draw(imageBLOBs)
        for BLOB in BLOBs:                
            y, x, r = BLOB          
            draw.ellipse((x-r, y-r, x+r, y+r), outline = (0, 225, 0))

        imagePlaceholder.image(imageBLOBs, use_column_width = True, caption = "Detected nanoparticles")
        st.session_state['imageBLOBs'] = imageBLOBs
    
    # Info about detected nanoparticles
    if st.session_state['detected']:
        st.markdown(f"<p class = 'text'> {st.session_state['BLOBs'].shape[0]} nanoparticles found! </p>", unsafe_allow_html=True)

    # Slider for comparing the results before and after detection
    if st.session_state['detected']:        
        st.checkbox("Comparison mode", key = 'comparison', help = help_str)

    # Saving
    if st.session_state['detected']:
        safeImgCol, safeBLOBCol = st.columns(2)

        with safeImgCol:
            temp = Image.new(mode="RGBA", size = st.session_state['imageBLOBs'].size)
            draw = ImageDraw.Draw(temp)
            for BLOB in st.session_state['BLOBs']:                
                y, x, r = BLOB          
                draw.ellipse((x-r, y-r, x+r, y+r), outline = (0, 225, 0))

            file = io.BytesIO()
            temp.save(file, format = "PNG")

            st.download_button(
                label = "Download image",
                data = file.getvalue(),
                file_name = "processed-image.tif",
                use_container_width  = True,
                help = help_str
            )

        with safeBLOBCol:
            file = io.StringIO()
            csv.writer(file).writerows(st.session_state['BLOBs'])

            st.download_button(
                label = "Download nanoparticles",
                data = file.getvalue(),
                file_name = "nanoparticles.csv",
                use_container_width  = True,
                help = help_str
            )
    
    # Nanoparticle mass
    if st.session_state['detected']:
        densityPd = 12.02 * 10**-15 # nanograms / nanometer        
        
        lowerBound = autoscale.findBorder(grayImage)
        
        flag = False
        if (lowerBound is not None):      
            text = autoscale.findText(grayImage[lowerBound:, :])
            print("Текст:", text)

            scaleVal = autoscale.scale(text)

            # Длина шкалы в пикселях
            scaleLengthVal = autoscale.scaleLength(grayImage, lowerBound)
            print(f"Длина шкалы: {scaleLengthVal} px")

            if (scaleVal is not None) and (scaleLengthVal is not None):
                print(f"nm / pixel: {scaleVal / scaleLengthVal}")
                print(f"pixel / nm: {scaleLengthVal / scaleVal}")        

                st.markdown(f"<p class = 'text'> Аutomatic scale calculation: <b>{scaleVal / scaleLengthVal:0.4} nm/px</b> </p>", unsafe_allow_html=True)

                radiusNM = st.session_state['BLOBs'][:, 2] * scaleVal / scaleLengthVal;
                V = 4 / 3 * np.pi * radiusNM ** 3
                massParticles = np.sum(V * densityPd)

                st.markdown(f"<p class = 'text'> Mass of detected Pd-nanoparticles is <b>{massParticles:0.2e} nanograms</b> </p>", unsafe_allow_html=True)
                flag = True
        
        if not flag:
            st.write(f"The image scale could not be determined automatically!")

            
st.markdown("<div class = 'footer'> Laboratory of Cognitive Technologies and Simulating Systems (LCTSS), Tula State University (TulSU) © 2024 </div>", unsafe_allow_html=True)